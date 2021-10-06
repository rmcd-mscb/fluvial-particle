"""ParticleTrack."""
# %%
import csv  # noqa
import os
from itertools import count

import h5py
import numpy as np
import vtk  # noqa
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
# %%
# HDF5 file writing initialization protocol
# In MPI, this whole section will need to be COLLECTIVE
vtkcoords = River.vtksgrid2d.GetPoints().GetData()
coords = numpy_support.vtk_to_numpy(vtkcoords)
x = coords[:, 0]
y = coords[:, 1]
x = x.reshape(ns, nn)
y = y.reshape(ns, nn)
dimtime = np.ceil(EndTime / (dt * print_inc))
file = h5py.File("results.h5", "w")
grpc = file.create_group("cells")
grpc["X"] = x
grpc["Y"] = y
grpc.create_dataset("FractionalParticleCount", (dimtime, ns - 1, nn - 1))
grpc.create_dataset("TotalParticleCount", (dimtime, ns - 1, nn - 1))
grpp = file.create_group("particles")
grpp.create_dataset("x", (dimtime, npart), dtype="f")
grpp.create_dataset("y", (dimtime, npart), dtype="f")
grpp.create_dataset("z", (dimtime, npart), dtype="f")
grpp.create_dataset("cell2D", (dimtime, npart), dtype="i")
grpp.create_dataset("cell3D", (dimtime, npart), dtype="i")
grpp.create_dataset("time", (dimtime, 1), dtype="f")
h5pyidx = 0
xmlfilename = "cells.xmf"
filexmf = open(xmlfilename, "w")
# Header for xml files
River.write_hdf5_xmf_header(filexmf)

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

    # Update the particle counts per cell
    """ np.add.at(NumPartIn3DCell, particles.cellindex3d, 1)
    np.add.at(NumPartInCell, particles.cellindex2d, 1)
    if np.sum(NumPartInCell) != npart:
        print("bad sum in NumPartInCell")
    CI_IDB = particles.cellindex2d % nsc
    np.add.at(PartInNSCellPTime, CI_IDB, 1)
    if np.sum(PartInNSCellPTime) != npart:
        print("bad sum in PartInNSCellPTime")
    np.add.at(PartTimeInCell, particles.cellindex2d, dt)
    np.add.at(TotPartInCell, particles.cellindex2d, 1) """

    # Print occasionally
    if count_index % print_inc == 0:
        # New HDF5 file writing protocol, saves iterates to same group as a temporal collection
        # River.write_hdf5(grpc, NumPartInCell / npart, h5pyidx, filexmf, TotTime)
        particles.write_hdf5(grpp, TotTime, h5pyidx)
        # particles.write_hdf5_xmf(filexmf2, TotTime, h5pyidx)
        h5pyidx = h5pyidx + 1
        # HDF5 printing end

        """ carray4 = vtk.vtkFloatArray()
        carray4.SetNumberOfValues(num2dcells)
        carray5 = vtk.vtkFloatArray()
        carray5.SetNumberOfValues(num3dcells)
        carray6 = vtk.vtkFloatArray()
        carray6.SetNumberOfValues(num2dcells)
        print(TotTime)
        t = count_index
        with open(next(g), "w") as tfile:
            writer = csv.writer(tfile)
            writer.writerow(
                (
                    "index",
                    "time",
                    "cellIndex",
                    "x",
                    "y",
                    "z",
                    "bed_elevation",
                    "Height_Above_Bed",
                    "WSE",
                )
            )
            # this may be wrong, but try it out
            (
                tt,
                cind,
                tx,
                ty,
                tz,
                telev,
                thtabvbed,
                twse,
            ) = particles.get_total_position()
            for p in anpart:  # need research on how to write full arrays
                writer.writerow(
                    (
                        p,
                        tt[p],
                        cind[p],
                        tx[p],
                        ty[p],
                        tz[p],
                        telev[p],
                        thtabvbed[p],
                        twse[p],
                    )
                )

        with open(next(gg), "w") as t2file:
            writer2 = csv.writer(t2file)
            writer2.writerow(("index", "NumPart"))
            for i_ind in range(nsc):
                writer2.writerow((i_ind, PartInNSCellPTime[i_ind]))
        # if Track2D:
        for n in range(num2dcells):
            carray4.SetValue(n, NumPartInCell[n] / npart)
        #     tmp = float(sum(partInCell[n,:]))
        #     # if tmp > 0:
        #     #     print tmp
        #     carray6.SetValue(n, tmp)
        River.vtksgrid2d.GetCellData().AddArray(carray4)
        carray4.SetName("Fractional Particle Count")
        # vtksgrid2d.GetCellData().AddArray(carray6)
        # carray6.SetName('Fract_Num_Particle')

        wsg = vtk.vtkStructuredGridWriter()
        wsg.SetInputData(River.vtksgrid2d)
        wsg.SetFileTypeToBinary()
        wsg.SetFileName(next(ggg))
        wsg.Write()
        River.vtksgrid2d.GetCellData().RemoveArray("Fractional Particle Count")
        # if Track3D:
        for n in range(num3dcells):
            carray5.SetValue(n, NumPartIn3DCell[n] / npart)
        River.vtksgrid3d.GetCellData().AddArray(carray5)
        carray5.SetName("Fractional Particle Count")
        wsg = vtk.vtkStructuredGridWriter()
        wsg.SetInputData(River.vtksgrid3d)
        wsg.SetFileTypeToBinary()
        wsg.SetFileName(next(g4))
        wsg.Write()
        River.vtksgrid3d.GetCellData().RemoveArray("Fractional Particle Count")


carray = vtk.vtkFloatArray()
carray.SetNumberOfValues(num2dcells)
carray2 = vtk.vtkFloatArray()
carray2.SetNumberOfValues(num2dcells)
carray3 = vtk.vtkFloatArray()
carray3.SetNumberOfValues(num2dcells)


for n in range(num2dcells):
    carray.SetValue(n, TotPartInCell[n])

    # Total time of particles in cell divided by
    # the number of particles in that cell
    if TotPartInCell[n] == 0:
        tmpval = 0
    else:
        tmpval = float(PartTimeInCell[n] / TotPartInCell[n])

    carray2.SetValue(n, tmpval)
    carray3.SetValue(n, float(PartTimeInCell[n] / EndTime))

River.vtksgrid2d.GetCellData().AddArray(carray)
carray.SetName("Particle Count in Cell")
River.vtksgrid2d.GetCellData().AddArray(carray2)
carray2.SetName("Avg Particle Time in Cell")
River.vtksgrid2d.GetCellData().AddArray(carray3)
carray3.SetName("Norm Particle Time in Cell")

wsg = vtk.vtkStructuredGridWriter()
wsg.SetInputData(River.vtksgrid2d)
wsg.SetFileTypeToBinary()
wsg.SetFileName("NoStrmLnCurv_185cms2d1_part.vtk")
wsg.Write() """

xmlfilename2 = "particles.xmf"
filexmf2 = open(xmlfilename2, "w")
# Header for xml files
particles.write_hdf5_xmf_header(filexmf2)
for i in range(h5pyidx):
    x = grpp["x"][i, :]
    y = grpp["y"][i, :]
    z = grpp["z"][i, :]
    cell2d = grpp["cell2D"][i, :]
    cell3d = grpp["cell3D"][i, :]
    time = grpp["time"][i]
    time = time.item(0)  # this returns a python scalar, for use in f-strings
    particles.write_hdf5_xmf(filexmf2, time, i)

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
    River.write_hdf5(grpc, name, NumPartInCell / npart, i, filexmf, time)


# Finalize HDF5/xmf file writing
River.write_hdf5_xmf_footer(filexmf)
particles.write_hdf5_xmf_footer(filexmf2)
filexmf.close()
filexmf2.close()
file.close()

if __name__ == "__main__":
    pass
