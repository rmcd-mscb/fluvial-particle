"""ParticleTrack."""
import os

import numpy as np

import fluvial_particle.settings as settings
from fluvial_particle.LarvalParticles import LarvalParticles  # noqa
from fluvial_particle.Particles import Particles  # noqa
from fluvial_particle.RiverGrid import RiverGrid

# from mpi4py import MPI


# Some Variables
# EndTime = 14400  # end time of simulation
EndTime = settings.SimTime
dt = 0.05  # dt of simulation
dt = settings.dt
min_depth = 0.01  # Minimum depth particles can enter]
min_depth = settings.min_depth

lev = settings.LEV  # lateral eddy viscosity

beta_x = 0.067
beta_y = 0.067
beta_z = 0.067

beta_x = settings.beta_x
beta_y = settings.beta_y
beta_z = settings.beta_z

# 2D or 3D particle tracking
Track2D = 0
Track3D = 1
Track2D = settings.Track2D
Track3D = settings.Track3D
print_inc = settings.PrintAtTick

# Fractional depth that bounds vertical particle positions from bed and WSE
if Track2D:
    alpha = 0.5
else:
    alpha = 0.01

# The source file
file_name_3da = r"C:\GitRepos\Python\ParticleTracking\Sum3_Result_3D_1.vtk"
file_name_2da = r"C:\GitRepos\Python\ParticleTracking\Sum3_Result_2D_1.vtk"
file_name_2da = settings.file_name_2da
file_name_3da = settings.file_name_3da

# Initialize RiverGrid object
River = RiverGrid(Track3D, file_name_2da, file_name_3da)
nsc = River.nsc
num3dcells = River.vtksgrid3d.GetNumberOfCells()
num2dcells = River.vtksgrid2d.GetNumberOfCells()
print(num3dcells, num2dcells)

# Initialize particles with initial location and attach RiverGrid
npart = 300  # number of particles per processor
npart = settings.NumPart

# # Get rank, number of processors, global number of particles, and local slice indices
# comm = MPI.COMM_WORLD
rank = 0  # comm.Get_rank()
size = 1  # comm.Get_size()
globalnparts = npart * size  # total number of particles across processors
start = rank * globalnparts  # slice starting index for HDF5 file
end = start + npart  # slice ending index (non-inclusive) for HDF5 file

xstart, ystart, zstart = settings.StartLoc
x = np.zeros(npart) + xstart
y = np.zeros(npart) + ystart
z = np.zeros(npart) + zstart
rng = np.random.default_rng(rank)
# confirmed in test.py this generates unique rands to each proc
# Other parallel random options available: https://numpy.org/devdocs/reference/random/parallel.html

# Sinusoid properties for larval drift subclass
amplitude = 1.0
period = 60.0
min_elev = 0.5
amplitude = settings.amplitude
period = settings.period
min_elev = settings.min_elev
ttime = rng.uniform(0.0, period, npart)
""" particles = LarvalParticles(
    npart, x, y, z, rng, River, 0.2, period, min_elev, ttime, Track2D, Track3D
) """

particles = Particles(npart, x, y, z, rng, River, Track2D, Track3D)
particles.initialize_location(0.5)  # 0.5 is midpoint of water column

TotTime = 0.0
count_index = 0
NumPartInCell = np.zeros(num2dcells, dtype=np.int64)
NumPartIn3DCell = np.zeros(num3dcells, dtype=np.int64)
PartTimeInCell = np.zeros(num2dcells)
TotPartInCell = np.zeros(num2dcells, dtype=np.int64)
PartInNSCellPTime = np.zeros(nsc, dtype=np.int64)

os.chdir(settings.out_dir)

# HDF5 file writing initialization protocol
# In MPI, this whole section will need to be COLLECTIVE
# Find total number of possible printing steps
dimtime = np.ceil(EndTime / (dt * print_inc)).astype("int")
# Create HDF5 particles dataset
parts_h5 = particles.create_hdf(dimtime, globalnparts)
# parts_h5 = Particles.create_hdf(dimtime, globalnparts, comm)  # MPI version
# end COLLECTIVE
# MPI Barrier

h5pyidx = 0
while TotTime <= EndTime:  # noqa C901
    # Increment counters, reset counter arrays
    TotTime = TotTime + dt
    count_index += 1
    print(TotTime, count_index)

    # Generate new random numbers
    particles.gen_rands()
    # Interpolate RiverGrid field data to particles
    particles.interp_fields
    # Calculate dispersion terms
    particles.calc_diffusion_coefs(lev, beta_x, beta_y, beta_z)
    # Move particles (checks on new position done internally)
    particles.move_all(alpha, min_depth, TotTime, dt)

    # Print occasionally
    if count_index % print_inc == 0:
        # INDEPENDENT write to HDF5
        # particles.write_hdf5(parts_h5, TotTime, h5pyidx)
        # MPI version:
        particles.write_hdf5(parts_h5, h5pyidx, start, end, TotTime, rank)
        h5pyidx += 1

# Write xml files and cumulative cell counters
# ROOT processor only
if rank == 0:
    # Create and open xdmf files
    cells1d_xmf = open("cells_onedim.xmf", "w")
    cells2d_xmf = open("cells_twodim.xmf", "w")
    cells3d_xmf = open("cells_threedim.xmf", "w")
    parts_xmf = open("particles.xmf", "w")
    River.write_hdf5_xmf_header1d(cells1d_xmf)
    River.write_hdf5_xmf_header2d(cells2d_xmf)
    River.write_hdf5_xmf_header3d(cells3d_xmf)
    particles.write_hdf5_xmf_header(parts_xmf)

    # Create cells HDF5 file
    grpc = parts_h5["coordinates"]
    grpp = parts_h5["properties"]
    time = grpc["time"]
    cells_h5 = River.create_hdf5(dimtime, time)

    # For every printing time loop, we load the particles data, sum the cell-centered counter arrays,
    # write the arrays to the cells HDF5, and write metadata to the XDMF files
    for i in range(h5pyidx):
        x = grpc["x"][i, :]
        y = grpc["y"][i, :]
        z = grpc["z"][i, :]
        t = time[i]
        cell2d = grpp["cellidx2d"][i, :]
        cell3d = grpp["cellidx3d"][i, :]
        t = t.item(0)  # this returns a python scalar, for use in f-strings
        particles.write_hdf5_xmf(parts_xmf, t, dimtime, npart, i)

        PartInNSCellPTime[:] = 0
        NumPartIn3DCell[:] = 0
        NumPartInCell[:] = 0
        CI_IDB = cell2d % nsc
        np.add.at(PartInNSCellPTime, CI_IDB, 1)
        np.add.at(PartTimeInCell, cell2d, dt)
        np.add.at(TotPartInCell, cell2d, 1)
        np.add.at(NumPartInCell, cell2d, 1)
        np.add.at(NumPartIn3DCell, cell3d, 1)

        # dims, name, and attrname must be passed to write_hdf5_xmf as iterable objects
        # dtypes too, but it is optional (defaults to "Float")
        name = [[]]
        attrname = [[]]
        name[0] = f"/cells1d/fpc{i}"
        attrname[0] = "FractionalParticleCount"
        data = PartInNSCellPTime / globalnparts
        River.write_hdf5(cells_h5, name[0], data)
        dims = (River.ns - 1,)
        River.write_hdf5_xmf(cells1d_xmf, t, dims, name, attrname, center="Node")

        name = [[], []]
        attrname = [[], []]
        dtypes = [[], []]
        name[0] = f"/cells2d/fpc{i}"
        attrname[0] = "FractionalParticleCount"
        dtypes[0] = "Float"
        dims = (River.ns - 1, River.nn - 1)
        data = (NumPartInCell / globalnparts).reshape(dims)
        River.write_hdf5(cells_h5, name[0], data)
        name[1] = f"/cells2d/tpc{i}"
        attrname[1] = "TotalParticleCount"
        dtypes[1] = "Int"
        dims = (River.ns - 1, River.nn - 1)
        data = TotPartInCell.reshape(dims)
        River.write_hdf5(cells_h5, name[1], data)
        River.write_hdf5_xmf(cells2d_xmf, t, dims, name, attrname, dtypes)

        name = [[]]
        attrname = [[]]
        name[0] = f"/cells3d/fpc{i}"
        attrname[0] = "FractionalParticleCount"
        dims = (River.ns - 1, River.nn - 1, River.nz - 1)
        data = (NumPartIn3DCell / globalnparts).reshape(dims)
        River.write_hdf5(cells_h5, name[0], data)
        River.write_hdf5_xmf(cells3d_xmf, t, dims, name, attrname)
    # Finalize xmf file writing
    River.write_hdf5_xmf_footer(cells1d_xmf)
    River.write_hdf5_xmf_footer(cells2d_xmf)
    River.write_hdf5_xmf_footer(cells3d_xmf)
    particles.write_hdf5_xmf_footer(parts_xmf)
    cells1d_xmf.close()
    cells2d_xmf.close()
    cells3d_xmf.close()
    parts_xmf.close()
    cells_h5.close()
# end ROOT section
# the preceeding section could be done on several processors, split over the first index (time)

# MPI Barrier

# COLLECTIVE file close
parts_h5.close()

if __name__ == "__main__":
    pass
