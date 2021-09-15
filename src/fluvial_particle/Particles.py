"""Particles Class module."""
import numpy as np
import vtk


class Particles:
    """A class of particles, each with a velocity, size, and mass."""

    def __init__(self, nparts, x, y, z, fielddata, track2d=0, track3d=1):
        """Initialize instance of class Particles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            fielddata (RiverGrid): class instance of the river hydrodynamic data
            track2d (bool): 1 if 2D model run, 0 else
            track3d (bool): 1 if 3D model run, 0 else
        """
        self.nparts = nparts
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.River = fielddata
        self.track2d = track2d
        self.track3d = track3d

        self.time = np.zeros(nparts)
        self.bedElev = np.zeros(nparts)
        self.htabvbed = np.zeros(nparts)
        self.wse = np.zeros(nparts)
        self.cellindex2d = np.zeros(nparts, dtype=np.int64)
        self.cellindex3d = np.zeros(nparts, dtype=np.int64)
        self.tmpelev = np.zeros(nparts)
        self.tmpwse = np.zeros(nparts)
        self.tmpdepth = np.zeros(nparts)
        self.tmpibc = np.zeros(nparts)
        # tmpvel = np.zeros(npart)
        self.tmpss = np.zeros(nparts)
        self.tmpvelx = np.zeros(nparts)
        self.tmpvely = np.zeros(nparts)
        self.tmpvelz = np.zeros(nparts)
        self.tmpustar = np.zeros(nparts)

    def setz(self, tz):
        """Set z-value.

        Args:
            tz (float): new z-value of particle
        """
        self.z = tz

    def move_all(self, vx, vy, vz, x_diff, y_diff, z_diff, xrnum, yrnum, zrnum, dt):
        """Update position based on speed, angle.

        Args:
            vx (float): flow velocity along the x axis
            vy (float): flow velocity along the y axis
            vz (float): flow velocity along the z axis
            x_diff (float): diffusion coefficient along x
            y_diff (float): diffusion coefficient along y
            z_diff (float): diffusion coefficient along z
            xrnum (float): random number from N(0,1), scales x diffusion
            yrnum (float): random number from N(0,1), scales y diffusion
            zrnum (float): random number from N(0,1), scales z diffusion
            dt (float): time step
        """
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = xrnum * (2.0 * x_diff * dt) ** 0.5
        yranwalk = yrnum * (2.0 * y_diff * dt) ** 0.5
        zranwalk = zrnum * (2.0 * z_diff * dt) ** 0.5
        # Move and update positions in-place on each array
        a = velmag > 0.0
        self.x[a] += (
            vx[a] * dt
            + ((xranwalk[a] * vx[a]) / velmag[a])
            - ((yranwalk[a] * vy[a]) / velmag[a])
        )
        self.y[a] += (
            vy[a] * dt
            + ((xranwalk[a] * vy[a]) / velmag[a])
            + ((yranwalk[a] * vx[a]) / velmag[a])
        )
        self.z = self.z + vz * dt + zranwalk

    def move_random_only_2d(self, x_diff, y_diff, xrnum, yrnum, boolarray, dt):
        """Update position based on random walk in x and y directions.

        Args:
            x_diff ([type]): [description]
            y_diff ([type]): [description]
            xrnum ([type]): [description]
            yrnum ([type]): [description]
            boolarray ([type]): [description]
            dt ([type]): [description]
        """
        self.x[boolarray] += xrnum[boolarray] * (2.0 * x_diff[boolarray] * dt) ** 0.5
        self.y[boolarray] += yrnum[boolarray] * (2.0 * y_diff[boolarray] * dt) ** 0.5

    def project_2d(self, vx, vy, x_diff, y_diff, xrnum, yrnum, dt):
        """Forward-project new 2D position based on speed, angle.

        Args:
            vx ([type]): [description]
            vy ([type]): [description]
            x_diff ([type]): [description]
            y_diff ([type]): [description]
            xrnum ([type]): [description]
            yrnum ([type]): [description]
            dt ([type]): [description]

        Returns:
            [type]: [description]
        """
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = xrnum * (2.0 * x_diff * dt) ** 0.5
        yranwalk = yrnum * (2.0 * y_diff * dt) ** 0.5
        px = np.copy(self.x)
        py = np.copy(self.y)

        a = velmag > 0.0
        px[a] += (
            vx[a] * dt
            + ((xranwalk[a] * vx[a]) / velmag[a])
            - ((yranwalk[a] * vy[a]) / velmag[a])
        )
        py[a] += (
            vy[a] * dt
            + ((xranwalk[a] * vy[a]) / velmag[a])
            + ((yranwalk[a] * vx[a]) / velmag[a])
        )
        return px, py

    def check_z(self, alpha, bedelev, wse):
        """[summary].

        Args:
            alpha ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]
        """
        # check on alpha? only makes sense for alpha<=0.5
        depth = wse - bedelev
        a = self.z > wse - alpha * depth
        b = self.z < bedelev + alpha * depth
        self.z[a] = wse[a] - alpha * depth[a]
        self.z[b] = bedelev[b] + alpha * depth[b]

    def update_info(self, cellind, time, bedelev, wse):
        """Update particle information."""
        self.bedElev = bedelev
        self.wse = wse
        self.htabvbed = self.z - self.bedElev
        self.time = time
        self.cellindex2d = cellind

    def get_total_position(self):
        """Return complete position of particle."""
        return (
            self.time,
            self.cellindex2d,
            self.x,
            self.y,
            self.z,
            self.bedElev,
            self.htabvbed,
            self.wse,
        )

    def get_cell_value(self, weights, idlist1, numpts, valarray):
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

    def get_cell_pos(self, newpoint2d, cellid):
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
        vtkcell2d = self.River.vtksgrid2d.GetCell(cellid)
        tmpres = vtkcell2d.EvaluatePosition(  # noqa F841
            newpoint2d, clspoint, tmpid, pcoords, vtkid2, weights
        )
        numpts = vtkcell2d.GetNumberOfPoints()
        idlist1 = vtkcell2d.GetPointIds()
        return weights, idlist1, numpts

    def get_vel3d_value(self, newpoint3d, cellid):
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
        vtkcell3d = self.River.vtksgrid3d.GetCell(cellid)
        result = vtkcell3d.EvaluatePosition(
            newpoint3d, clspoint, tmpid, pcoords, vtkid2, weights
        )
        numpts = vtkcell3d.GetNumberOfPoints()
        idlist1 = vtkcell3d.GetPointIds()
        tmpxval = np.float64(0.0)
        tmpyval = np.float64(0.0)
        for i in range(0, numpts):
            tmpxval += (
                weights[i] * self.River.VelocityVec3D.GetTuple(idlist1.GetId(i))[0]
            )
            tmpyval += (
                weights[i] * self.River.VelocityVec3D.GetTuple(idlist1.GetId(i))[1]
            )
        return result, vtkid2, tmpxval, tmpyval, 0.0

    def get_vel2d_value(self, weights, idlist1, numpts):
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
            tmpxval += (
                weights[i] * self.River.VelocityVec2D.GetTuple(idlist1.GetId(i))[0]
            )
            tmpyval += (
                weights[i] * self.River.VelocityVec2D.GetTuple(idlist1.GetId(i))[1]
            )
        return tmpxval, tmpyval

    def interpolate_fields(self, count_index):
        """[Summary]."""
        point2d = np.vstack((self.x, self.y, np.zeros(self.nparts))).T
        for i in range(self.nparts):
            self.cellindex2d[i] = self.River.CellLocator2D.FindCell(point2d[i, :])
        # if np.any(self.cellindex2d < 0):
        #    print("initial cell -1")  # untested
        # Get information from vtk 2D grids for each particle
        for i in range(self.nparts):
            weights, idlist1, numpts = self.get_cell_pos(
                point2d[i, :], self.cellindex2d[i]
            )
            self.tmpelev[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.Elevation_2D
            )
            self.tmpwse[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.WSE_2D
            )
            self.tmpdepth[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.Depth_2D
            )
            self.tmpibc[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.IBC_2D
            )
            # tmpvel[i] = get_cell_value(weights, idlist1, numpts, Velocity_2D)
            self.tmpss[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.ShearStress2D
            )
            if ~self.track3d:
                self.tmpvelx[i], self.tmpvely[i] = self.get_vel2d_value(
                    weights, idlist1, numpts
                )
        # check shear stress (without error print statements)
        self.tmpss = np.where(self.tmpss < 0.0, 0.0, self.tmpss)
        self.tmpustar = (self.tmpss / 1000.0) ** 0.5

        # MOVE ELSEWHERE; Check particle depths and calc PartNormDepth
        if count_index <= 1:
            self.check_z(0.5, self.tmpelev, self.tmpwse)

        if self.track3d:
            self.PartNormDepth = (self.z - self.tmpelev) / self.tmpdepth

        # Get 3D Velocity Components
        if self.track3d:
            self.calc_3dfields_at_nodes()

    def calc_3dfields_at_nodes(self):
        """[Summary]."""
        # Locate particle in 3D grid and interpolate velocity
        idlist1 = vtk.vtkIdList()
        for i in range(self.nparts):
            point3d = [self.x[i], self.y[i], self.z[i]]
            self.cellindex3d[i] = self.River.CellLocator3D.FindCell(point3d)
            if self.cellindex3d[i] >= 0:
                (
                    result,
                    t_dist,
                    t_tmp3dux,
                    t_tmp3duy,
                    t_tmp3duz,
                ) = self.get_vel3d_value(point3d, self.cellindex3d[i])
                self.tmpvelx[i] = t_tmp3dux
                self.tmpvely[i] = t_tmp3duy
                self.tmpvelz[i] = t_tmp3duz
            else:
                print("3d findcell failed, particle number: ", i)
                print("switching to FindCellsAlongLine() method")
                pp1 = [point3d[0], point3d[1], self.tmpwse[i] + 10]
                pp2 = [point3d[0], point3d[1], self.tmpelev[i] - 10]
                self.River.CellLocator3D.FindCellsAlongLine(pp1, pp2, 0.0, idlist1)
                maxdist = 1e6
                for t in range(idlist1.GetNumberOfIds()):
                    (
                        result,
                        t_dist,
                        t_tmp3dux,
                        t_tmp3duy,
                        t_tmp3duz,
                    ) = self.get_vel3d_value(point3d, idlist1.GetId(t))
                    if result == 1:
                        self.tmpvelx[i] = t_tmp3dux
                        self.tmpvely[i] = t_tmp3duy
                        self.tmpvelz[i] = t_tmp3duz
                        self.cellindex3d[i] = idlist1.GetId(t)
                        break
                    elif t_dist < maxdist:
                        maxdist = t_dist
                        self.tmpvelx[i] = t_tmp3dux
                        self.tmpvely[i] = t_tmp3duy
                        self.tmpvelz[i] = t_tmp3duz
                        self.cellindex3d[i] = idlist1.GetId(t)
                if self.cellindex3d[i] < 0:
                    print("part still out of 3d grid")
                    self.tmpvelx[i] = 0.0
                    self.tmpvely[i] = 0.0
                    self.tmpvelz[i] = 0.0
                """print("3d findcell failed, particle number: ", i)
                print("Particle location: ", point3d_2[i, :])
                print("Particle fractional depth: ", PartNormDepth[i])
                print("closest 3D cell, 2Dcell: ", cellid3d[i], cellid[i])
                vtkcell = vtksgrid2d.GetCell(cellid[i])
                vtkptlist = vtkcell.GetPointIds()
                vtkpts = vtkcell.GetPoints()
                print("2D grid points:")
                for j in range(vtkcell.GetNumberOfPoints()):
                print(vtkpts.GetPoint(j))
                print(Elevation_2D.GetTuple(vtkptlist.GetId(j)))
                vtkcell = vtksgrid3d.GetCell(cellid3d[i])
                vtkpts = vtkcell.GetPoints()
                print("3D grid points:")
                for j in range(vtkpts.GetNumberOfPoints()):
                print(vtkpts.GetPoint(j)) """

    def is_cell_wet_helper(self, weights, idlist1, numpts):
        """[summary].

        Args:
            weights ([type]): [description]
            idlist1 ([type]): [description]
            numpts ([type]): [description]

        Returns:
            [type]: [description]
        """
        tmpibc = 0.0
        for i in range(numpts):
            tmpibc += weights[i] * self.River.IBC_2D.GetTuple(idlist1.GetId(i))[0]
        if tmpibc >= 0.9999999:
            return True
        else:
            return False

    def is_cell_wet(self, px, py):
        """[summary].

        Args:
            px ([type]): [description]
            py ([type]): [description]

        Returns:
            [type]: [description]
        """
        newpoint2d = np.vstack((px, py, np.zeros(self.nparts))).T
        cellidb = np.zeros(self.nparts, dtype=np.int64)
        wet = np.empty(self.nparts, dtype=bool)
        for i in range(self.nparts):
            cellidb[i] = self.River.CellLocator2D.FindCell(newpoint2d[i, :])
            weights, idlist1, numpts = self.get_cell_pos(newpoint2d[i, :], cellidb[i])
            wet[i] = self.is_cell_wet_helper(weights, idlist1, numpts)
        return cellidb, wet
