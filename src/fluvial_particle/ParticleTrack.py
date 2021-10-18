"""ParticleTrack."""
import os

import numpy as np

import fluvial_particle.settings as settings
from fluvial_particle.LarvalParticles import LarvalParticles  # noqa
from fluvial_particle.Particles import Particles  # noqa
from fluvial_particle.RiverGrid import RiverGrid


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
npart = 300  # number of particles
npart = settings.NumPart

# # Determine rank, number of processors, and distribution of particles to processors
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# # https://stackoverflow.com/questions/15658145/how-to-share-work-roughly-evenly-between-processes-in-mpi-despite-the-array-size/26554699  # noqa
# count = globalnparts // size  # integer division
# remainder = globalnparts % size
# if (rank < remainder):
#   start = rank * (count + 1)
#   end = start + count + 1  # non-inclusive ending index
# else:
#   start = rank * count + remainder
#   end = start + count
# nparts = end - start
start = 0
end = npart
rank = 0  # for now, so I can implement start, end, and rank in h5py methods

xstart, ystart, zstart = settings.StartLoc
x = np.zeros(npart) + xstart
y = np.zeros(npart) + ystart
z = np.zeros(npart) + zstart
rng = np.random.default_rng(rank)
# For MPI version, seed with rank? otherwise they'll all get the random same #s
# Other parallel options: https://numpy.org/devdocs/reference/random/parallel.html

# Sinusoid properties for larval drift subclass
amplitude = 1.0
period = 60.0
min_elev = 0.5
amplitude = settings.amplitude
period = settings.period
min_elev = settings.min_elev
ttime = rng.uniform(0.0, period, npart)  # not parallel compatible ?

particles = Particles(npart, x, y, z, rng, River, Track2D, Track3D)
""" particles = LarvalParticles(
    npart, x, y, z, rng, River, 0.2, period, min_elev, ttime, Track2D, Track3D
) """
# Particles start at midpoint of water column
particles.initialize_location(0.9)

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
parts_h5 = particles.create_hdf(dimtime, npart)
# parts_h5 = Particles.create_hdf(dimtime, globalnparts)  # MPI version
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
        particles.write_hdf5(parts_h5, TotTime, h5pyidx, start, end, rank)
        h5pyidx = h5pyidx + 1

# Write xml files and cumulative cell counters
# ROOT processor only
if rank == 0:
    cells_h5 = River.create_hdf5(dimtime)
    cells_xmf = open("cells.xmf", "w")
    parts_xmf = open("particles.xmf", "w")
    River.write_hdf5_xmf_header(cells_xmf)
    particles.write_hdf5_xmf_header(parts_xmf)
    for i in range(h5pyidx):
        x = parts_h5["coords/x"][i, :]
        y = parts_h5["coords/y"][i, :]
        z = parts_h5["coords/z"][i, :]
        cell2d = parts_h5["cell2d"][i, :]
        cell3d = parts_h5["cell3d"][i, :]
        time = parts_h5["time"][i]
        time = time.item(0)  # this returns a python scalar, for use in f-strings
        particles.write_hdf5_xmf(parts_xmf, time, dimtime, npart, i)

        PartInNSCellPTime[:] = 0
        NumPartIn3DCell[:] = 0
        NumPartInCell[:] = 0
        np.add.at(NumPartInCell, cell2d, 1)
        np.add.at(NumPartIn3DCell, cell3d, 1)
        CI_IDB = cell2d % nsc
        np.add.at(PartInNSCellPTime, CI_IDB, 1)
        np.add.at(PartTimeInCell, cell2d, dt)
        np.add.at(TotPartInCell, cell2d, 1)

        # ADD 3d fpc, part time in cell, part in cell along streamline, totpartincell
        name = "FractionalParticleCount"
        River.write_hdf5(cells_h5, name, NumPartInCell / npart, i)
        River.write_hdf5_xmf(cells_xmf, time, dimtime, name, cells_h5[name].name, i)
    # Finalize xmf file writing
    River.write_hdf5_xmf_footer(cells_xmf)
    particles.write_hdf5_xmf_footer(parts_xmf)
    cells_xmf.close()
    parts_xmf.close()
    cells_h5.close()
# end ROOT section
# the preceeding section could be done on several processors, split over the first index (time)

# MPI Barrier

# COLLECTIVE file close
parts_h5.close()

if __name__ == "__main__":
    pass
