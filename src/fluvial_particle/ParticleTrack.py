"""ParticleTrack."""
# %%
import csv
import os
from itertools import count

import numpy as np
import vtk

import fluvial_particle.settings as settings
from fluvial_particle.Particles import Particles
from fluvial_particle.RiverGrid import RiverGrid


def gen_filenames(prefix, suffix, places=3):
    """Generate sequential filenames with the format <prefix><index><suffix>.

    The index field is padded with leading zeroes to the specified number of places

    http://stackoverflow.com/questions/5068461/how-do-you-increment-file-name-in-python
    """
    pattern = "{}{{:0{}d}}{}".format(prefix, places, suffix)
    for i in count(1):
        yield pattern.format(i)


def get_vel3d_value(newpoint3d, cellid):
    """[summary].

    Args:
        newpoint3d ([type]): [description]
        cellid ([type]): [description]

    Returns:
        [type]: [description]
    """
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    clspoint = [0.0, 0.0, 0.0]
    tmpid = vtk.mutable(0)
    vtkid2 = vtk.mutable(0)
    vtkcell3d = River.vtksgrid3d.GetCell(cellid)
    result = vtkcell3d.EvaluatePosition(
        newpoint3d, clspoint, tmpid, pcoords, vtkid2, weights
    )
    # print result, clspoint, tmpid, pcoords, vtkid2
    idlist1 = vtk.vtkIdList()
    numpts = vtkcell3d.GetNumberOfPoints()
    idlist1 = vtkcell3d.GetPointIds()
    tmpxval = np.float64(0.0)
    tmpyval = np.float64(0.0)
    for i in range(0, numpts):
        tmpxval += weights[i] * VelocityVec3D.GetTuple(idlist1.GetId(i))[0]
        tmpyval += weights[i] * VelocityVec3D.GetTuple(idlist1.GetId(i))[1]
    return result, vtkid2, tmpxval, tmpyval, 0.0


def get_vel2d_value(weights, idlist1, numpts):
    """Get 2D velocity vector value from nodes in idlist1 given weights.

    Args:
        weights ([type]): [description]
        idlist1 ([type]): [description]
        numpts ([type]): [description]

    Returns:
        [type]: [description]
    """
    tmpxval = np.float64(0.0)
    tmpyval = np.float64(0.0)
    for i in range(numpts):
        tmpxval += weights[i] * VelocityVec2D.GetTuple(idlist1.GetId(i))[0]
        tmpyval += weights[i] * VelocityVec2D.GetTuple(idlist1.GetId(i))[1]
    return tmpxval, tmpyval


def get_cell_value(weights, idlist1, numpts, valarray):
    """Get value from given array from nodes in idlist1 given weights.

    Args:
        weights ([type]): [description]
        idlist1 ([type]): [description]
        numpts ([type]): [description]
        valarray ([type]): [description]

    Returns:
        [type]: [description]
    """
    tmpval = np.float64(0.0)
    for i in range(numpts):
        tmpval += weights[i] * valarray.GetTuple(idlist1.GetId(i))[0]

    return tmpval


def is_cell_wet(weights, idlist1, numpts):
    """[summary].

    Args:
        weights ([type]): [description]
        idlist1 ([type]): [description]
        numpts ([type]): [description]

    Returns:
        [type]: [description]
    """
    tmpibc = 0.0
    for x in range(0, numpts):
        tmpibc += weights[x] * IBC_2D.GetTuple(idlist1.GetId(x))[0]
    if tmpibc >= 0.9999999:
        return True
    else:
        return False


def get_cell_pos(newpoint2d, cellid):
    """[summary].

    Args:
        newpoint2d ([type]): [description]
        cellid ([type]): [description]

    Returns:
        [type]: [description]
    """
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    clspoint = [0.0, 0.0, 0.0]
    tmpid = vtk.mutable(0)
    vtkid2 = vtk.mutable(0)
    vtkcell2d = River.vtksgrid2d.GetCell(cellid)
    tmpres = vtkcell2d.EvaluatePosition(  # noqa F841
        newpoint2d, clspoint, tmpid, pcoords, vtkid2, weights
    )
    numpts = vtkcell2d.GetNumberOfPoints()
    idlist1 = vtkcell2d.GetPointIds()
    return weights, idlist1, numpts


# Some Variables
# EndTime = 14400  # end time of simulation
EndTime = settings.SimTime
dt = 0.05  # dt of simulation
dt = settings.dt
# avg_shear_dev = 0.14      # reach averaged value of dispersion
avg_shear_dev = 0.01  # for simple meander Tut5
avg_shear_dev = settings.avg_shear_dev
avg_bed_shearstress = settings.avg_bed_shearstress
avg_depth = settings.avg_depth
avg_shear_dev = 0.067 * avg_depth * np.sqrt(avg_bed_shearstress / 1000)
min_depth = 0.01  # Minimum depth particles can enter]
min_depth = settings.min_depth

vert_type = settings.vert_type

beta_x = 0.067
beta_y = 0.067
beta_z = 0.067

beta_x = settings.beta_x
beta_y = settings.beta_y
beta_z = settings.beta_z

avg_shear_devx = beta_x * avg_depth * np.sqrt(avg_bed_shearstress / 1000.0)
avg_shear_devy = beta_y * avg_depth * np.sqrt(avg_bed_shearstress / 1000.0)
avg_shear_devz = beta_z * avg_depth * np.sqrt(avg_bed_shearstress / 1000.0)

# 2D or 3D particle tracking
Track2D = 1
Track3D = 1
Track2D = settings.Track2D
Track3D = settings.Track3D
print_inc = settings.PrintAtTick

# Fractional depth that bounds particle positions from bed and WSE
if Track2D:
    alpha = 0.5
    PartNormDepth = 0.5
else:
    alpha = 0.01

# The source file
file_name_3da = r"C:\GitRepos\Python\ParticleTracking\Sum3_Result_3D_1.vtk"
file_name_2da = r"C:\GitRepos\Python\ParticleTracking\Sum3_Result_2D_1.vtk"
file_name_2da = settings.file_name_2da
file_name_3da = settings.file_name_3da

# vtkSolName = r'C:\GitRepos\Python\Results\Arcrom_Sum_River\Sim6\Result_2D_1.vtk'

# River_Mile = "G:\IPC\Data\RiverMile\rivermile.csv"
# rmData = []

# with open(r"G:\IPC\Data\RiverMile\rivermile.csv", 'rt') as rmf:
#     reader = csv.reader(rmf)
#     tcount = 0
#     for row in reader:
#         tcount += 1
#         if tcount > 1:
#             rmData.append(row)

# print(file_name_3da)
# print(file_name_2da)

# Initialize RiverGrid object
River = RiverGrid(Track3D)
River.read_2d_data(file_name_2da)
River.read_3d_data(file_name_3da)
River.load_arrays()
River.build_locators()

CellLocator3D = River.CellLocator3D
CellLocator2D = River.CellLocator2D
WSE_2D = River.WSE_2D
Depth_2D = River.Depth_2D
Elevation_2D = River.Elevation_2D
# Velocity_2D = vtksgrid2d.GetPointData().GetScalars("Velocity (magnitude)")
IBC_2D = River.IBC_2D
VelocityVec2D = River.VelocityVec2D
ShearStress2D = River.ShearStress2D
VelocityVec3D = River.VelocityVec3D
num3dcells = River.vtksgrid3d.GetNumberOfCells()
num2dcells = River.vtksgrid2d.GetNumberOfCells()
print(num3dcells, num2dcells)

ns = River.ns
nn = River.nn
nz = River.nz
nsc = River.nsc
nnc = River.nnc

# add Particles
npart = 300  # number of particles
npart = settings.NumPart
xstart, ystart, zstart = settings.StartLoc

# Initialize particles class with initial location and attach RiverGrid
x = np.zeros(npart) + xstart
y = np.zeros(npart) + ystart
z = np.zeros(npart) + zstart
particles = Particles(npart, x, y, z, River, Track2D, Track3D)
# particles_last = Particles(npart, x, y, z, River)
rng = np.random.default_rng(0)  # Numpy recommended method for new code
anpart = np.arange(npart).tolist()

# Sinusoid properties; these aren't used, REMOVE?
# Will be used for larval drift subclass eventually
amplitude = 1.0
period = 60.0
min_elev = 0.5
amplitude = settings.amplitude
period = settings.period
min_elev = settings.min_elev
ttime = rng.uniform(0.0, period, npart)

TotTime = 0.0
count_index = 0
NumPartInCell = np.zeros(num2dcells, dtype=np.int64)
NumPartIn3DCell = np.zeros(num3dcells, dtype=np.int64)
# partInCell = np.zeros((num2dcells,npart), dtype = int)
PartTimeInCell = np.zeros(num2dcells)
TotPartInCell = np.zeros(num2dcells, dtype=np.int64)
PartInNSCellPTime = np.zeros(nsc, dtype=np.int64)
tmpelev = np.zeros(npart)
tmpwse = np.zeros(npart)
tmpdepth = np.zeros(npart)
tmpibc = np.zeros(npart)
# tmpvel = np.zeros(npart)
tmpss = np.zeros(npart)
tmpvelx = np.zeros(npart)
tmpvely = np.zeros(npart)
tmpvelz = np.zeros(npart)
os.chdir(settings.out_dir)
g = gen_filenames("fish1_", ".csv")
gg = gen_filenames("nsPart_", ".csv")
ggg = gen_filenames("Sim_Result_2D_", ".vtk")
g4 = gen_filenames("Sim_Result_3D_", ".vtk")
while TotTime <= EndTime:  # noqa C901
    # Increment counters
    TotTime = TotTime + dt
    count_index += 1
    PartInNSCellPTime[:] = 0
    NumPartIn3DCell[:] = 0
    NumPartInCell[:] = 0
    print(TotTime, count_index)

    # Get random numbers
    xrnum = rng.standard_normal(npart)
    yrnum = rng.standard_normal(npart)
    if Track2D:
        zrnum = np.zeros(npart)
    else:
        zrnum = rng.standard_normal(npart)

    # Interpolate RiverGrid field data to particles
    particles.interpolate_fields(count_index)

    tmpelev = particles.tmpelev
    tmpwse = particles.tmpwse
    tmpdepth = particles.tmpdepth
    tmpibc = particles.tmpibc
    # tmpvel = np.zeros(npart)
    tmpss = particles.tmpss
    tmpvelx = particles.tmpvelx
    tmpvely = particles.tmpvely
    tmpvelz = particles.tmpvelz
    tmpustar = particles.tmpustar
    PartNormDepth = particles.PartNormDepth
    for i in anpart:
        if particles.cellindex3d[i] >= 0:
            NumPartIn3DCell[particles.cellindex3d[i]] += 1

    # Calculate dispersion terms
    ustarh = (tmpwse - tmpelev) * tmpustar
    Dx = settings.LEV + beta_x * ustarh
    Dy = settings.LEV + beta_y * ustarh
    Dz = settings.LEV + beta_z * ustarh

    # Update particle positions
    # First, forward-project to new (x,y) coordinates
    p2x, p2y = particles.project_2d(tmpvelx, tmpvely, Dx, Dy, xrnum, yrnum, dt)
    # Second, get boolean array from is_cell_wet
    newpoint2d = np.vstack((p2x, p2y, np.zeros(npart))).T
    cellidb = np.zeros(npart, dtype=np.int64)
    wet1 = np.empty(npart, dtype=bool)
    for i in anpart:
        cellidb[i] = CellLocator2D.FindCell(newpoint2d[i, :])
        weights, idlist1, numpts = get_cell_pos(newpoint2d[i, :], cellidb[i])
        wet1[i] = is_cell_wet(weights, idlist1, numpts)
    if np.any(~wet1):
        # print("dry cell encountered")
        # Third, forward-project dry cells using just random motion
        p2x[~wet1] = particles.x[~wet1] + xrnum[~wet1] * (2.0 * Dx[~wet1] * dt) ** 0.5
        p2y[~wet1] = particles.y[~wet1] + yrnum[~wet1] * (2.0 * Dy[~wet1] * dt) ** 0.5
        # Fourth, run is_cell_wet again
        newpoint2d = np.vstack((p2x, p2y, np.zeros(npart))).T
        wet2 = np.empty(npart, dtype=bool)
        for i in anpart:
            cellidb[i] = CellLocator2D.FindCell(newpoint2d[i, :])
            weights, idlist1, numpts = get_cell_pos(newpoint2d[i, :], cellidb[i])
            wet2[i] = is_cell_wet(weights, idlist1, numpts)
        # Fifth, any still dry entries will have zero positional update this step
        p2x[~wet2] = particles.x[~wet2]
        p2y[~wet2] = particles.y[~wet2]
        # Sixth, manually update (x,y) of particles that were not wet first time, yes wet second time
        a = ~wet1 & wet2
        particles.move_random_only_2d(Dx, Dy, xrnum, yrnum, a, dt)
        tmpvelx[~wet1] = 0.0  # ensure that move_all() does nothing for these particles
        tmpvely[~wet1] = 0.0
        tmpvelz[~wet1] = 0.0
        Dx[~wet1] = 0.0
        Dy[~wet1] = 0.0
        Dz[~wet1] = 0.0
        newpoint2d = np.vstack((p2x, p2y, np.zeros(npart))).T
        for i in anpart:
            cellidb[i] = CellLocator2D.FindCell(newpoint2d[i, :])
    # Seventh, prevent particles from entering cells where tdepth1 < min_depth
    elev1 = np.zeros(npart)
    wse1 = np.zeros(npart)
    for i in anpart:
        weights, idlist1, numpts = get_cell_pos(newpoint2d[i, :], cellidb[i])
        elev1[i] = get_cell_value(weights, idlist1, numpts, Elevation_2D)
        wse1[i] = get_cell_value(weights, idlist1, numpts, WSE_2D)
    tdepth1 = wse1 - elev1
    b = tdepth1 < min_depth
    if np.any(b):
        print("particle entered min_depth")
        p2x[b] = particles.x[b]
        p2y[b] = particles.y[b]
        tmpvelx[b] = 0.0
        tmpvely[b] = 0.0
        tmpvelz[b] = 0.0
        Dx[b] = 0.0
        Dy[b] = 0.0
        Dz[b] = 0.0
        # Eighth, update vertical position to same fractional depth as last
        newpoint2d = np.vstack((p2x, p2y, np.zeros(npart))).T
        for i in anpart:
            cellidb[i] = CellLocator2D.FindCell(newpoint2d[i, :])
            weights, idlist1, numpts = get_cell_pos(newpoint2d[i, :], cellidb[i])
            elev1[i] = get_cell_value(weights, idlist1, numpts, Elevation_2D)
            wse1[i] = get_cell_value(weights, idlist1, numpts, WSE_2D)
    CI_IDB = cellidb % nsc
    tdepth1 = wse1 - elev1
    p2z = elev1 + (PartNormDepth * tdepth1)
    particles.setz(p2z)
    # Ninth, run particles.move_all
    particles.move_all(tmpvelx, tmpvely, tmpvelz, Dx, Dy, Dz, xrnum, yrnum, zrnum, dt)
    # Tenth, final check that new coords are all within vertical domain
    particles.check_z(alpha, elev1, wse1)

    # Update location information
    particles.update_info(cellidb, np.full(npart, TotTime), elev1, wse1)

    # Update the particle counts per cell
    np.add.at(NumPartInCell, cellidb, 1)
    if np.sum(NumPartInCell) != npart:
        print("bad sum in NumPartInCell")
    np.add.at(PartInNSCellPTime, CI_IDB, 1)
    if np.sum(PartInNSCellPTime) != npart:
        print("bad sum in PartInNSCellPTime")
    np.add.at(PartTimeInCell, cellidb, dt)
    np.add.at(TotPartInCell, cellidb, 1)

    carray4 = vtk.vtkFloatArray()
    carray4.SetNumberOfValues(num2dcells)
    carray5 = vtk.vtkFloatArray()
    carray5.SetNumberOfValues(num3dcells)
    carray6 = vtk.vtkFloatArray()
    carray6.SetNumberOfValues(num2dcells)

    # for t in range(count_index):
    if count_index % print_inc == 0:
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
    carray.SetValue(n, NumPartInCell[n])

    # Total time of particles in cell divided by
    # the number of particles in that cell
    if NumPartInCell[n] == 0:
        tmpval = 0
    else:
        tmpval = float(PartTimeInCell[n] / NumPartInCell[n])

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
wsg.Write()

if __name__ == "__main__":
    pass

# %%
