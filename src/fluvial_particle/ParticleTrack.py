"""ParticleTrack."""
# %%
import os
from itertools import count

import h5py
import numpy as np
from vtk.util import numpy_support  # type:ignore

import fluvial_particle.settings as settings
from fluvial_particle.LarvalParticles import LarvalParticles  # noqa
from fluvial_particle.Particles import Particles  # noqa
from fluvial_particle.RiverGrid import RiverGrid


def gen_filenames(prefix, suffix, places=3):
    """Generate sequential filenames with the format <prefix><index><suffix>.

    The index field is padded with leading zeroes to the specified number of places

    http://stackoverflow.com/questions/5068461/how-do-you-increment-file-name-in-python
    """
    pattern = "{}{{:0{}d}}{}".format(prefix, places, suffix)
    for i in count(1):
        yield pattern.format(i)


# Some Variables
# EndTime = 14400  # end time of simulation
EndTime = settings.SimTime
dt = 0.05  # dt of simulation
dt = settings.dt
# avg_shear_dev = 0.14      # reach averaged value of dispersion
avg_shear_dev = 0.01  # not used
avg_shear_dev = settings.avg_shear_dev  # not used
avg_bed_shearstress = settings.avg_bed_shearstress  # not used
avg_depth = settings.avg_depth  # not used
avg_shear_dev = 0.067 * avg_depth * np.sqrt(avg_bed_shearstress / 1000)  # not used
min_depth = 0.01  # Minimum depth particles can enter]
min_depth = settings.min_depth

vert_type = settings.vert_type  # not used

lev = settings.LEV  # lateral eddy viscosity

beta_x = 0.067
beta_y = 0.067
beta_z = 0.067

beta_x = settings.beta_x
beta_y = settings.beta_y
beta_z = settings.beta_z

avg_shear_devx = beta_x * avg_depth * np.sqrt(avg_bed_shearstress / 1000.0)  # not used
avg_shear_devy = beta_y * avg_depth * np.sqrt(avg_bed_shearstress / 1000.0)  # not used
avg_shear_devz = beta_z * avg_depth * np.sqrt(avg_bed_shearstress / 1000.0)  # not used

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
ns = River.ns  # not used
nn = River.nn  # not used
nz = River.nz  # not used
nsc = River.nsc
nnc = River.nnc  # not used
num3dcells = River.vtksgrid3d.GetNumberOfCells()
num2dcells = River.vtksgrid2d.GetNumberOfCells()
print(num3dcells, num2dcells)

# Initialize particles with initial location and attach RiverGrid
npart = 300  # number of particles
npart = settings.NumPart
xstart, ystart, zstart = settings.StartLoc
x = np.zeros(npart) + xstart
y = np.zeros(npart) + ystart
z = np.zeros(npart) + zstart
rng = np.random.default_rng(0)  # Numpy recommended method for new code

# Sinusoid properties for larval drift subclass
amplitude = 1.0
period = 60.0
min_elev = 0.5
amplitude = settings.amplitude
period = settings.period
min_elev = settings.min_elev
ttime = rng.uniform(0.0, period, npart)

particles = Particles(npart, x, y, z, rng, River, Track2D, Track3D)
""" particles = LarvalParticles(
    npart, x, y, z, rng, River, 0.2, period, min_elev, ttime, Track2D, Track3D
) """
# Particles start at midpoint of water column
particles.initialize_location(0.9)
anpart = np.arange(npart).tolist()

TotTime = 0.0
count_index = 0
NumPartInCell = np.zeros(num2dcells, dtype=np.int64)
NumPartIn3DCell = np.zeros(num3dcells, dtype=np.int64)
# partInCell = np.zeros((num2dcells,npart), dtype = int)
PartTimeInCell = np.zeros(num2dcells)
TotPartInCell = np.zeros(num2dcells, dtype=np.int64)
PartInNSCellPTime = np.zeros(nsc, dtype=np.int64)

os.chdir(settings.out_dir)
g = gen_filenames("fish1_", ".csv")
gg = gen_filenames("nsPart_", ".csv")
ggg = gen_filenames("Sim_Result_2D_", ".vtk")
g4 = gen_filenames("Sim_Result_3D_", ".vtk")

# HDF5 file writing initialization protocol
# In MPI, this whole section will need to be COLLECTIVE
vtkcoords = River.vtksgrid2d.GetPoints().GetData()
coords = numpy_support.vtk_to_numpy(vtkcoords)
x = coords[:, 0]
y = coords[:, 1]
x = x.reshape(ns, nn)
y = y.reshape(ns, nn)
dimtime = np.ceil(EndTime / (dt * print_inc)).astype("int")
arr = np.zeros((dimtime, ns - 1, nn - 1))
cells_h5 = h5py.File("cells.h5", "w")
parts_h5 = h5py.File("particles.h5", "w")
cells_h5.create_dataset("X", (ns, nn), dtype="f", data=x)
cells_h5.create_dataset("Y", (ns, nn), dtype="f", data=y)
cells_h5.create_dataset("FractionalParticleCount", (dimtime, ns - 1, nn - 1), data=arr)
parts_h5.create_dataset("x", (dimtime, npart), dtype="f")
parts_h5.create_dataset("y", (dimtime, npart), dtype="f")
parts_h5.create_dataset("z", (dimtime, npart), dtype="f")
parts_h5.create_dataset("bedelev", (dimtime, npart), dtype="f")
parts_h5.create_dataset("htabvbed", (dimtime, npart), dtype="f")
parts_h5.create_dataset("wse", (dimtime, npart), dtype="f")
parts_h5.create_dataset("velvec", (dimtime, npart, 3), dtype="f")
parts_h5.create_dataset("cell2D", (dimtime, npart), dtype="i")
parts_h5.create_dataset("cell3D", (dimtime, npart), dtype="i")
parts_h5.create_dataset("time", (dimtime, 1), dtype="f")
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
        # New HDF5 file writing protocol, saves iterates to same group as a temporal collection
        particles.write_hdf5(parts_h5, TotTime, h5pyidx)
        h5pyidx = h5pyidx + 1

# Write xml files and cumulative cell counters
cells_xmf = open("cells.xmf", "w")
parts_xmf = open("particles.xmf", "w")
River.write_hdf5_xmf_header(cells_xmf)
particles.write_hdf5_xmf_header(parts_xmf)
for i in range(h5pyidx):
    x = parts_h5["x"][i, :]
    y = parts_h5["y"][i, :]
    z = parts_h5["z"][i, :]
    cell2d = parts_h5["cell2D"][i, :]
    cell3d = parts_h5["cell3D"][i, :]
    time = parts_h5["time"][i]
    time = time.item(0)  # this returns a python scalar, for use in f-strings
    particles.write_hdf5_xmf(parts_xmf, time, dimtime, i)

    PartInNSCellPTime[:] = 0
    NumPartIn3DCell[:] = 0
    NumPartInCell[:] = 0
    np.add.at(NumPartInCell, cell2d, 1)
    np.add.at(NumPartIn3DCell, cell3d, 1)
    CI_IDB = cell2d % nsc
    np.add.at(PartInNSCellPTime, CI_IDB, 1)
    np.add.at(PartTimeInCell, cell2d, dt)
    np.add.at(TotPartInCell, cell2d, 1)

    name = "FractionalParticleCount"
    River.write_hdf5(cells_h5, name, NumPartInCell / npart, i)
    River.write_hdf5_xmf(cells_xmf, time, dimtime, name, cells_h5[name].name, i)

# Finalize HDF5/xmf file writing
River.write_hdf5_xmf_footer(cells_xmf)
particles.write_hdf5_xmf_footer(parts_xmf)
cells_xmf.close()
parts_xmf.close()
cells_h5.close()
parts_h5.close()

if __name__ == "__main__":
    pass
