"""ParticleTrack."""
import csv
import math
import os
import random
from itertools import count

import numpy as np
import vtk
from vtk.util import numpy_support  # type:ignore

import fluvial_particle.settings as settings
from fluvial_particle.Particles import Particles


def dist(x1, y1, x2, y2, x3, y3):  # x3,y3 is the point
    """http://www.autohotkey.com/board/topic/60656-calculate-the-distance-between-a-point-and-a-line-segment/."""
    px = x2 - x1
    py = y2 - y1

    something = (px * px) + (py * py)

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt((dx * dx) + (dy * dy))
    return dist, x, y


def gen_filenames(prefix, suffix, places=3):
    """Generate sequential filenames with the format <prefix><index><suffix>.

    The index field is padded with leading zeroes to the specified number of places

    http://stackoverflow.com/questions/5068461/how-do-you-increment-file-name-in-python
    """
    pattern = "{}{{:0{}d}}{}".format(prefix, places, suffix)
    for i in count(1):
        yield pattern.format(i)


def get_3d_vec_value(newpoint3d, cellid):
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
    vtkcell3d = vtk.vtkHexahedron()
    vtkcell3d = vtksgrid3d.GetCell(cellid)
    result = vtkcell3d.EvaluatePosition(
        newpoint3d, clspoint, tmpid, pcoords, vtkid2, weights
    )
    # print result, clspoint, tmpid, pcoords, vtkid2
    idlist1 = vtk.vtkIdList()
    numpts = vtkcell3d.GetNumberOfPoints()
    idlist1 = vtkcell3d.GetPointIds()
    tmpxval = 0.0
    tmpyval = 0.0
    for x in range(0, numpts):
        tmpxval = tmpxval + weights[x] * VelocityVec3D.GetTuple(idlist1.GetId(x))[0]
        tmpyval = tmpyval + weights[x] * VelocityVec3D.GetTuple(idlist1.GetId(x))[1]
    return result, vtkid2, tmpxval, tmpyval, 0.0


def get_2d_vec_value(newpoint2d, cellid):
    """Get 2D vector value at point and cellid.

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
    vtkcell2d = vtk.vtkQuad()
    vtkcell2d = vtksgrid2d.GetCell(cellid)
    tmpres = vtkcell2d.EvaluatePosition(  # noqa F841
        newpoint2d, clspoint, tmpid, pcoords, vtkid2, weights
    )
    idlist1 = vtk.vtkIdList()
    numpts = vtkcell2d.GetNumberOfPoints()
    idlist1 = vtkcell2d.GetPointIds()
    tmpxval = 0.0
    tmpyval = 0.0
    for x in range(0, numpts):
        tmpxval = tmpxval + weights[x] * VelocityVec2D.GetTuple(idlist1.GetId(x))[0]
        tmpyval = tmpyval + weights[x] * VelocityVec2D.GetTuple(idlist1.GetId(x))[1]
    return tmpxval, tmpyval


def get_cell_value(newpoint2d, cellid, valarray):
    """Get cell value at point and cellid with given array.

    Args:
        newpoint2d ([type]): [description]
        cellid ([type]): [description]
        valarray ([type]): [description]

    Returns:
        [type]: [description]
    """
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    clspoint = [0.0, 0.0, 0.0]
    tmpid = vtk.mutable(0)
    vtkid2 = vtk.mutable(0)
    vtkcell2d = vtk.vtkQuad()
    vtkcell2d = vtksgrid2d.GetCell(cellid)
    tmpres = vtkcell2d.EvaluatePosition(  # noqa F841
        newpoint2d, clspoint, tmpid, pcoords, vtkid2, weights
    )
    idlist1 = vtk.vtkIdList()
    numpts = vtkcell2d.GetNumberOfPoints()
    idlist1 = vtkcell2d.GetPointIds()
    tmpval = 0.0
    for x in range(0, numpts):
        tmpval = tmpval + weights[x] * valarray.GetTuple(idlist1.GetId(x))[0]

    return tmpval


def is_cell_wet(newpoint2d, cellid):
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
    vtkcell2d = vtk.vtkQuad()
    vtkcell2d = vtksgrid2d.GetCell(cellid)
    tmpres = vtkcell2d.EvaluatePosition(  # noqa F841
        newpoint2d, clspoint, tmpid, pcoords, vtkid2, weights
    )
    idlist1 = vtk.vtkIdList()
    numpts = vtkcell2d.GetNumberOfPoints()
    idlist1 = vtkcell2d.GetPointIds()
    tmpibc = 0.0
    for x in range(0, numpts):
        tmpibc = tmpibc + weights[x] * IBC_2D.GetTuple(idlist1.GetId(x))[0]
    if tmpibc >= 0.9999999:
        return True
    else:
        return False


def get_reflected_position(cellid, point1, point2):
    """[summary].

    Args:
        cellid ([type]): [description]
        point1 ([type]): [description]
        point2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    vtkcell2d = vtk.vtkQuad()
    vtkcell2d = vtksgrid2d.GetCell(cellid)
    numedges = vtkcell2d.GetNumberOfEdges()
    points = vtk.vtkPoints()
    found = False
    for e in range(numedges):
        npx = npy = 0.0
        points = vtkcell2d.GetEdge(e).GetPoints()
        e1 = points.GetPoint(0)
        e2 = points.GetPoint(1)
        e1p = (e1[0], e1[1])
        e2p = (e2[0], e2[1])
        line1 = (e1p, e2p)
        p1 = (point1[0], point1[1])
        p2 = (point2[0], point2[1])
        line2 = (p1, p2)
        success = False
        success, ipx, ipy = line_intersection(line1, line2)

        if success:
            if (min(point1[0], point2[0]) < ipx < max(point1[0], point2[0])) and (
                min(point1[1], point2[1]) < ipy < max(point1[1], point2[1])
            ):
                if point1[0] < ipx:
                    dx = point2[0] - ipx
                elif point1[0] > ipx:
                    dx = point1[0] - ipx
                else:
                    dx = 0.0
                if point1[1] < ipy:
                    dy = point2[1] - ipy
                elif point1[1] > ipy:
                    dy = point1[1] - ipy
                else:
                    dy = 0.0

                npx = ipx + dx
                npy = ipy + dy
                found = True
    if found:
        return found, npx, npy
    else:
        return found, npx, npy


def line_intersection(line1, line2):
    """http://stackoverflow.com/questions/20677795/find-the-point-of-intersecting-lines."""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False, -1.0, -1.0

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return True, x, y


def calc_grid_metrics(xx, yy):  # noqa
    """[summary].

    Args:
        xx ([type]): [description]
        yy ([type]): [description]

    Returns:
        [type]: [description]
    """
    nn, ns = xx.shape
    jmid = int((nn + 1) / 2)

    stot = 0.0
    for i in range(1, ns):
        ds = np.sqrt(
            np.power(xx[jmid, i] - xx[jmid, i - 1], 2)
            + np.power(yy[jmid, i] - yy[jmid, i - 1], 2)
        )
        stot += ds
    print(stot)

    scals = stot / (ns - 1)
    xshift = xx[jmid, 0]
    yshift = yy[jmid, 0]

    xo = np.zeros(ns)
    yo = np.zeros(ns)
    phirotation = np.zeros(ns)
    phi = np.zeros(ns)
    rn = np.zeros((nn, ns))
    r = np.zeros(ns)

    tmpwidth = np.sqrt(
        np.power(xx[0, 0] - xx[nn - 1, 0], 2) + np.power(yy[0, 0] - yy[nn - 1, 0], 2)
    )
    for i in range(ns):
        xo[i] = xx[jmid, i] - xshift
        yo[i] = yy[jmid, i] - yshift

    slin = np.sqrt(np.power(xo[ns - 1], 2) + np.power(yo[ns - 1], 2))
    fcos = xo[ns - 1] / slin
    fsin = yo[ns - 1] / slin

    mo = stot
    wmax = tmpwidth
    dn = wmax / (nn - 1)
    ds = mo / (ns - 1)
    nm = int((nn + 1) / 2.0)

    print(mo, wmax, dn, ds, nm)

    for i in range(ns):
        xc1 = xo[i] * fcos + yo[i] * fsin
        yc1 = yo[i] * fcos - xo[i] * fsin
        xo[i] = xc1
        yo[i] = yc1

    for i in range(1, ns):
        dx = xo[i] - xo[i - 1]
        dy = yo[i] - yo[i - 1]
        if dx == 0:
            if dy > 0:
                phirotation[i] = np.arccos(-1.0) / 2.0
            else:
                phirotation[i] = -1.0 * np.arccos(-1.0) / 2.0
        else:
            phirotation[i] = np.arctan2(dy, dx)
        phi[i] = phirotation[i]
    phirotation[0] = (2.0 * phirotation[1]) - phirotation[2]
    phi[0] = phirotation[1]

    for i in range(1, ns):
        dphi = phirotation[i] - phirotation[i - 1]
        if dphi == 0:
            if r[i - 1] < 0:
                r[i] = -1000000.0
            else:
                r[i] = 1000000.0
        else:
            r[i] = scals / dphi

    r[0] = (2.0 * r[1]) - r[2]

    nm = (nn + 1) / 2
    for i in range(ns):
        for j in range(nn):
            rn[j, i] = 1.0 - ((j + 1) - nm) * dn / r[i]
    print(r)
    return fcos, fsin, ds, dn, phirotation, rn


random.seed()  # remove this in favor of np.random ?
np.random.seed()
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
avg_shear_dev = (
    0.067 * settings.avg_depth * np.sqrt(settings.avg_bed_shearstress / 1000)
)
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

amplitude = 1.0
period = 60.0
min_elev = 0.5

amplitude = settings.amplitude
period = settings.period
min_elev = settings.min_elev

# 2D or 3D particle tracking
Track2D = 1
Track3D = 1

TrackwDrift = 0

Track2D = settings.Track2D
Track3D = settings.Track3D

TrackwDrift = settings.TrackwDrift

# Grid xstart, ystart, zstart = settings.StartLoc
x_offset = 0.0  # x coordinate offset
y_offset = 0.0  # ycoordinate offset

print_inc = settings.PrintAtTick

# add Particles
npart = 300  # number of particles
npart = settings.NumPart
xstart, ystart, zstart = settings.StartLoc

# Initialize particles class with initial location
x = np.zeros(npart, dtype=float) + xstart  # broadcasting correctly?
y = np.zeros(npart, dtype=float) + ystart
z = np.zeros(npart, dtype=float) + zstart
ttime = np.random.uniform(0.0, period, npart)
particles = Particles(npart, x, y, z, ttime, amplitude, period, min_elev)

# Initialize particles with initial location
# for i in range(npart):
#    ttime = random.uniform(0, period)  # noqa S311
#    particle = Particle(i,(920.585, -161.145, 66.29), ttime, amplitude, period, min_elev)
#    particle = Particle(i, xstart, ystart, zstart, ttime, amplitude, period, min_elev)

#    particles.append(particle)

# px,py,pz = particles[0].get_position()
# print px, py, pz

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

print(file_name_3da)
print(file_name_2da)

# Read the 3Dsource file.
vtksgrid3d = vtk.vtkStructuredGrid()
reader3d = vtk.vtkStructuredGridReader()
reader3d.SetFileName(file_name_3da)
reader3d.SetOutput(vtksgrid3d)
reader3d.Update()  # Needed because of GetScalarRange
output3d = reader3d.GetOutput()
scalar_range = output3d.GetScalarRange()

print(vtksgrid3d)


# Read the 2Dsource file.
vtksgrid2d = vtk.vtkStructuredGrid()
reader2d = vtk.vtkStructuredGridReader()
reader2d.SetFileName(file_name_2da)
reader2d.SetOutput(vtksgrid2d)
reader2d.Update()  # Needed because of GetScalarRange
output2d = reader2d.GetOutput()
scalar_range = output2d.GetScalarRange()

num3dcells = vtksgrid3d.GetNumberOfCells()
num2dcells = vtksgrid2d.GetNumberOfCells()

print(num3dcells, num2dcells)


ns, nn, nz = vtksgrid2d.GetDimensions()
nsc = ns - 1
nnc = nn - 1

NumPartInCell = np.zeros((num2dcells), dtype=float)
NumPartIn3DCell = np.zeros((num3dcells), dtype=float)
# partInCell = np.zeros((num2dcells,npart), dtype = int)
PartTimeInCell = np.zeros((num2dcells), dtype=float)
TotPartInCell = np.zeros((num2dcells), dtype=float)
PartInNSCellPTime = np.zeros((nsc), dtype=int)


CellLocator3D = vtk.vtkCellLocator()
CellLocator3D.SetDataSet(vtksgrid3d)
# CellLocator3D.SetNumberOfCellsPerBucket(5);
CellLocator3D.SetTolerance(0.000000001)
CellLocator3D.BuildLocator()

CellLocator2D = vtk.vtkCellLocator()
CellLocator2D.SetDataSet(vtksgrid2d)
CellLocator2D.SetNumberOfCellsPerBucket(5)
CellLocator2D.BuildLocator()

# Get Elevation and WSE from 2D Grid
WSE_2D = vtksgrid2d.GetPointData().GetScalars("WaterSurfaceElevation")
Depth_2D = vtksgrid2d.GetPointData().GetScalars("Depth")
Elevation_2D = vtksgrid2d.GetPointData().GetScalars("Elevation")
Velocity_2D = vtksgrid2d.GetPointData().GetScalars("Velocity (magnitude)")
IBC_2D = vtksgrid2d.GetPointData().GetScalars("IBC")
VelocityVec2D = vtksgrid2d.GetPointData().GetVectors("Velocity")
ShearStress2D = vtksgrid2d.GetPointData().GetScalars("ShearStress (magnitude)")
# Get Velocity from 3D
VelocityVec3D = vtksgrid3d.GetPointData().GetScalars("Velocity")

# Calculate the gradient of the main dispersion terms depth*ustar
WSE_2D_np = numpy_support.vtk_to_numpy(WSE_2D)
WSE_2D_np = WSE_2D_np.reshape(nn, ns)
IBC_2D_np = numpy_support.vtk_to_numpy(IBC_2D)
IBC_2D_np = IBC_2D_np.reshape(nn, ns)
Depth_2D_np = numpy_support.vtk_to_numpy(Depth_2D)
Depth_2D_np = Depth_2D_np.reshape(nn, ns)
ShearStress_2D_np = numpy_support.vtk_to_numpy(ShearStress2D)
ShearStress_2D_np = ShearStress_2D_np.reshape(nn, ns)
coords = vtksgrid2d.GetPoints().GetData()
# print coords
np_coords = numpy_support.vtk_to_numpy(coords)
# print np_coords
xx = np_coords[:, 0]
yy = np_coords[:, 1]
# print xx.shape
xx = xx.reshape((nn, ns))
yy = yy.reshape((nn, ns))

TotTime = 0.0
count_index = 0

os.chdir(settings.out_dir)
g = gen_filenames("fish1_", ".csv")
gg = gen_filenames("nsPart_", ".csv")
ggg = gen_filenames("Sim_Result_2D_", ".vtk")
g4 = gen_filenames("Sim_Result_3D_", ".vtk")
while TotTime <= EndTime:  # noqa C901
    TotTime = TotTime + dt
    count_index += 1
    PartInNSCellPTime[:] = 0
    NumPartIn3DCell[:] = 0
    NumPartInCell[:] = 0
    print(TotTime, count_index)
    for n in range(npart):
        # get random numbers
        xrnum = random.gauss(0.0, 1.0)
        yrnum = random.gauss(0.0, 1.0)
        zrnum = random.gauss(0.0, 1.0)

        # Find particles 2D position in 2DCell
        px, py, pz = particles[n].get_position()
        Point2D = [px, py, 0.0]
        Point3D = [px, py, pz]
        cellid = CellLocator2D.FindCell(Point2D)
        if cellid < 0:
            print("initial cell -1")
        CI_ID = cellid % nsc  # this is the cell along the centerline of the grid

        tmpelev = get_cell_value(Point2D, cellid, Elevation_2D)
        tmpwse = get_cell_value(Point2D, cellid, WSE_2D)
        tmpdepth = get_cell_value(Point2D, cellid, Depth_2D)
        tmpibc = get_cell_value(Point2D, cellid, IBC_2D)
        tmpvel = get_cell_value(Point2D, cellid, Velocity_2D)
        tmpss = get_cell_value(Point2D, cellid, ShearStress2D)
        tmpvelx, tmpvely = get_2d_vec_value(Point2D, cellid)

        if pz < tmpelev:
            print("Error z < bedelev")
            pz = tmpelev + 0.5 * tmpdepth

        if tmpss < 0:
            print(
                f"error: ustar < 0, elev: {tmpelev}, depth: {tmpdepth}, wse: {tmpwse}, ibc: {tmpibc}, shear: {tmpss}"
            )
            tmpss = 0.0

        tmpustar = math.sqrt(tmpss / 1000.0)
        # Set particle 0.5 meters above bed
        if count_index <= 1:  # initialize starting depth of particles
            # pz = tmpelev+0.5
            if pz > tmpwse:
                pz = tmpelev + 0.5 * tmpdepth
        else:
            if pz >= tmpwse - 0.025 * (tmpdepth):
                pz = tmpwse - 0.025 * (tmpdepth)
                # print('part above surface')
            elif pz <= tmpelev + 0.025 * tmpdepth:
                pz = tmpelev + 0.025 * tmpdepth
                # print("part below bed")

        PartNormDepth = (pz - tmpelev) / tmpdepth
        Point3D = [px, py, pz]
        # Get3D Velocity Components
        tmpcell = vtk.vtkGenericCell()
        tpcoords = [0.0, 0.0, 0.0]
        tweights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        idlist1 = vtk.vtkIdList()
        pp1 = [px, py, tmpwse + 10]
        pp2 = [px, py, tmpelev - 10]
        tmp3duz = tmp3duy = tmp3dux = 0.0
        CellLocator3D.FindCellsAlongLine(pp1, pp2, 0.0, idlist1)
        CellId3D = 0
        maxdist = 1e6
        # print idlist1.GetNumberOfIds()
        for t in range(0, idlist1.GetNumberOfIds()):
            result, t_dist, t_tmp3dux, t_tmp3duy, t_tmp3duz = get_3d_vec_value(
                Point3D, idlist1.GetId(t)
            )
            if result == 1:
                tmp3dux = t_tmp3dux
                tmp3duy = t_tmp3duy
                tmp3duz = t_tmp3duz
                CellId3D = idlist1.GetId(t)
                #               print n, result, CellId3D, tmp3dux, tmp3duy, tmp3duz
                break
            elif t_dist < maxdist:
                maxdist = t_dist
                tmp3dux = t_tmp3dux
                tmp3duy = t_tmp3duy
                tmp3duz = t_tmp3duz
                CellId3D = idlist1.GetId(t)
                # print n, result, CellId3D, tmp3dux, tmp3duy, tmp3duz
        if CellId3D == 0:
            print(n, count_index, "no 3dcell found")
            CellId3D = 0

        if CellId3D < 0:
            print("part out of 3d grid")
            print(
                n,
                count_index,
                pz,
                tmpelev,
                tmpwse,
                (tmpwse - tmpelev),
                tmp3dux,
                tmp3duy,
            )
            tmp3dux = 0.0
            tmp3duy = 0.0
            tmp3duz = 0.0

        else:
            NumPartIn3DCell[CellId3D] += 1

        Dx = beta_x * (tmpwse - tmpelev) * tmpustar
        Dy = beta_y * (tmpwse - tmpelev) * tmpustar
        Dz = beta_z * (tmpwse - tmpelev) * tmpustar

        if settings.DispersionType == 1:
            Dx = beta_x * (tmpwse - tmpelev) * tmpustar
            Dy = beta_y * (tmpwse - tmpelev) * tmpustar
            Dz = beta_z * (tmpwse - tmpelev) * tmpustar

        elif settings.DispersionType == 2:

            Dx = settings.LEV + (beta_x * (tmpwse - tmpelev) * tmpustar)
            Dy = settings.LEV + (beta_y * (tmpwse - tmpelev) * tmpustar)
            Dz = settings.LEV + (beta_z * (tmpwse - tmpelev) * tmpustar)

        elif settings.DispersionType == 3:
            Dx = avg_shear_devx
            Dy = avg_shear_devy
            Dz = avg_shear_devz

        elif settings.DispersionType == 4:
            Dx = avg_shear_devx + settings.LEV
            Dy = avg_shear_devy + settings.LEV
            Dz = avg_shear_devz + settings.LEV
        #   Get new location of particle
        if Track2D:
            p2x, p2y, p2z = particles[n].move(
                count_index, tmpvelx, tmpvely, 0.0, Dx, Dy, xrnum, yrnum, dt
            )
        else:
            p2x, p2y, p2z = particles[n].move(
                count_index, tmp3dux, tmp3duy, 0.0, Dx, Dy, xrnum, yrnum, dt
            )
        newpoint2d = [p2x, p2y, 0.0]
        # check: inWetCell?

        #         newpoint3d = [p2x, p2y, p2z]

        # Check if new position is in wet or dry cell
        cellidb = CellLocator2D.FindCell(newpoint2d)
        if cellidb < 0:
            print("cellidB error")

        elev1 = get_cell_value(newpoint2d, cellidb, Elevation_2D)
        wse1 = get_cell_value(newpoint2d, cellidb, WSE_2D)
        tdepth1 = wse1 - elev1
        p2z = elev1 + (PartNormDepth * tdepth1)
        particles[n].setz(p2z)

        if p2z <= elev1:
            print("error pt <= elev")
        if p2z >= wse1:
            print("error pt >= wse")

        CI_IDB = cellidb % nsc
        if is_cell_wet(newpoint2d, cellidb):
            elev2 = get_cell_value(newpoint2d, cellidb, Elevation_2D)
            wse2 = get_cell_value(newpoint2d, cellidb, WSE_2D)
            #             if elev2>wse2:
            #                 print 'Error elev > wse'
            if Track2D:
                p2z = particles[n].vert_mean_depth(TotTime, elev2, wse2)
            else:
                if vert_type == 0:
                    p2z = particles[n].vert_const_depth(TotTime, elev2, wse2)
                elif vert_type == 1:
                    p2z = particles[n].vert_sinusoid(TotTime, elev2, wse2)
                elif vert_type == 2:
                    p2z = particles[n].vert_sinusoid_bottom(TotTime, elev2, wse2, 0.2)
                elif vert_type == 3:
                    p2z = particles[n].vert_sinusoid_surface(TotTime, elev2, wse2, 0.2)
                elif vert_type == 4:
                    p2z = particles[n].vert_sawtooth(TotTime, elev2, wse2)
                elif vert_type == 5:
                    p2z = particles[n].vert_random_walk(
                        TotTime, elev2, wse2, tmp3dux, tmp3duy, 0.0, Dz, zrnum, dt
                    )
                else:
                    print("vert_type not defined")

            if (
                wse2 - elev2
            ) < min_depth:  # Don't allow to move into depths less than min_depth
                #                 particles[n].update_position(count_index, px, py, pz, TotTime, tmpelev, tmpwse)
                particles[n].keep_postition(TotTime)
                NumPartInCell[cellid] += 1
                PartTimeInCell[cellid] += dt
                PartInNSCellPTime[CI_ID] += 1
                print("In Min Depth")
            else:
                particles[n].update_position(
                    count_index, cellidb, p2x, p2y, p2z, TotTime, elev2, wse2
                )
                NumPartInCell[cellidb] += 1
                PartTimeInCell[cellidb] += dt
                PartInNSCellPTime[CI_IDB] += 1
        else:
            prx, pry, prz = particles[n].move_random_only_2d(
                count_index, avg_shear_dev, avg_shear_dev, xrnum, yrnum, dt
            )

            newpoint2db = [prx, pry, 0.0]
            cellidc = CellLocator2D.FindCell(newpoint2db)
            CI_IDC = cellidc % nsc
            if is_cell_wet(newpoint2db, cellidc):
                elev2b = get_cell_value(newpoint2db, cellid, Elevation_2D)
                wse2b = get_cell_value(newpoint2db, cellid, WSE_2D)

                if Track2D:
                    p2zb = particles[n].vert_mean_depth(TotTime, elev2b, wse2b)
                else:
                    if vert_type == 0:
                        p2zb = particles[n].vert_const_depth(TotTime, elev2b, wse2b)
                    elif vert_type == 1:
                        p2zb = particles[n].vert_sinusoid(TotTime, elev2b, wse2b)
                    elif vert_type == 2:
                        p2zb = particles[n].vert_sinusoid_bottom(
                            TotTime, elev2b, wse2b, 0.2
                        )
                    elif vert_type == 3:
                        p2zb = particles[n].vert_sinusoid_surface(
                            TotTime, elev2b, wse2b, 0.2
                        )
                    elif vert_type == 4:
                        p2zb = particles[n].vert_sawtooth(TotTime, elev2b, wse2b)
                    elif vert_type == 5:
                        p2zb = particles[n].vert_random_walk(
                            TotTime, elev2b, wse2b, tmp3dux, tmp3duy, 0.0, Dz, zrnum, dt
                        )
                    else:
                        print("vert_type not defined")

                if (
                    wse2b - elev2b
                ) < min_depth:  # Don't allow to move into depths less than min_depth
                    particles[n].update_position(
                        count_index, cellid, px, py, pz, TotTime, tmpelev, tmpwse
                    )

                    NumPartInCell[cellid] += 1
                    PartTimeInCell[cellid] += dt
                    PartInNSCellPTime[CI_ID] += 1
                #                     print 'cell wet min_depth'
                else:
                    particles[n].update_position(
                        count_index, cellidc, prx, pry, p2zb, TotTime, elev2b, wse2b
                    )

                    NumPartInCell[cellidc] += 1
                    PartTimeInCell[cellidc] += dt
                    PartInNSCellPTime[CI_IDC] += 1
            #                     print 'cell wet new position'
            else:

                particles[n].keep_postition(TotTime)
                NumPartInCell[cellid] += 1
                PartTimeInCell[cellid] += dt
                PartInNSCellPTime[CI_ID] += 1
                tmppt = [px, py, 0.0]
                cellidd = CellLocator2D.FindCell(tmppt)
                print("cell wet old pos")

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
            for p in range(npart):
                tind, tt, cind, tx, ty, tz, telev, thtabvbed, twse = particles[
                    p
                ].get_total_position()
                writer.writerow((tind, tt, cind, tx, ty, tz, telev, thtabvbed, twse))

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
        vtksgrid2d.GetCellData().AddArray(carray4)
        carray4.SetName("Fractional Particle Count")
        # vtksgrid2d.GetCellData().AddArray(carray6)
        # carray6.SetName('Fract_Num_Particle')

        wsg = vtk.vtkStructuredGridWriter()
        wsg.SetInputData(vtksgrid2d)
        wsg.SetFileTypeToBinary()
        wsg.SetFileName(next(ggg))
        wsg.Write()
        vtksgrid2d.GetCellData().RemoveArray("Fractional Particle Count")
        # if Track3D:
        for n in range(num3dcells):
            carray5.SetValue(n, NumPartIn3DCell[n] / npart)
        vtksgrid3d.GetCellData().AddArray(carray5)
        carray5.SetName("Fractional Particle Count")
        wsg = vtk.vtkStructuredGridWriter()
        wsg.SetInputData(vtksgrid3d)
        wsg.SetFileTypeToBinary()
        wsg.SetFileName(next(g4))
        wsg.Write()
        vtksgrid3d.GetCellData().RemoveArray("Fractional Particle Count")


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

vtksgrid2d.GetCellData().AddArray(carray)
carray.SetName("Particle Count in Cell")
vtksgrid2d.GetCellData().AddArray(carray2)
carray2.SetName("Avg Particle Time in Cell")
vtksgrid2d.GetCellData().AddArray(carray3)
carray3.SetName("Norm Particle Time in Cell")

wsg = vtk.vtkStructuredGridWriter()
wsg.SetInputData(vtksgrid2d)
wsg.SetFileTypeToBinary()
wsg.SetFileName("NoStrmLnCurv_185cms2d1_part.vtk")
wsg.Write()

if __name__ == "__main__":
    pass
