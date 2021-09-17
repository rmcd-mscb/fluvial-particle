"""Particles Class module."""
import numpy as np
import vtk


class Particles:
    """A class of particles, each with a velocity, size, and mass."""

    def __init__(self, nparts, x, y, z, rng, fielddata, track2d=0, track3d=1):
        """Initialize instance of class Particles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            fielddata (RiverGrid): class instance of the river hydrodynamic data
            track2d (bool): 1 if 2D model run, 0 else
            track3d (bool): 1 if 3D model run, 0 else
        """
        self.nparts = nparts
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.rng = rng
        self.River = fielddata
        self.track2d = track2d
        self.track3d = track3d
        # Add an XOR on track2d & track3d ?

        self.time = np.zeros(nparts)
        self.bedElev = np.zeros(nparts)
        self.htabvbed = np.zeros(nparts)
        self.wse = np.zeros(nparts)
        self.cellindex2d = np.zeros(nparts, dtype=np.int64)
        self.cellindex3d = np.zeros(nparts, dtype=np.int64)
        self.depth = np.zeros(nparts)
        self.shearstress = np.zeros(nparts)
        self.velx = np.zeros(nparts)
        self.vely = np.zeros(nparts)
        self.velz = np.zeros(nparts)
        self.ustar = np.zeros(nparts)
        self.Dx = np.zeros(nparts)
        self.Dy = np.zeros(nparts)
        self.Dz = np.zeros(nparts)
        self.xrnum = np.zeros(nparts)
        self.yrnum = np.zeros(nparts)
        self.zrnum = np.zeros(nparts)
        # tmpibc = np.zeros(nparts)
        # tmpvel = np.zeros(npart)

    def setz(self, tz):
        """Set z-value.

        Args:
            tz (float): new z-value of particle
        """
        self.z = tz

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

    def move_all(self, dt):
        """Update position based on speed, angle.

        Args:
            dt (float): time step
        """
        vx = self.velx
        vy = self.vely
        vz = self.velz
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = self.xrnum * (2.0 * self.Dx * dt) ** 0.5
        yranwalk = self.yrnum * (2.0 * self.Dy * dt) ** 0.5
        zranwalk = self.zrnum * (2.0 * self.Dz * dt) ** 0.5
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

    def move_random_only_2d(self, boolarray, dt):
        """Update position based on random walk in x and y directions.

        Args:
            boolarray ([type]): [description]
            dt ([type]): [description]
        """
        self.x[boolarray] += (
            self.xrnum[boolarray] * (2.0 * self.Dx[boolarray] * dt) ** 0.5
        )
        self.y[boolarray] += (
            self.yrnum[boolarray] * (2.0 * self.Dy[boolarray] * dt) ** 0.5
        )

    def project_2d(self, min_depth, dt):
        """Forward-project new 2D position based on speed, angle.

        Args:
            min_depth ([type]): [description]
            dt ([type]): [description]

        Returns:
            [type]: [description]
        """
        vx = self.velx
        vy = self.vely
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = self.xrnum * (2.0 * self.Dx * dt) ** 0.5
        yranwalk = self.yrnum * (2.0 * self.Dy * dt) ** 0.5
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

        wet1 = self.is_cell_wet(px, py)
        if np.any(~wet1):
            self.handle_dry_parts(wet1, px, py, dt)

        # Prevent particles from entering cells where tdepth1 < min_depth
        elev1, wse1 = self.prevent_mindepth(px, py, min_depth)
        # Update particle elevation in new water column to same fractional depth as last
        tdepth1 = wse1 - elev1
        p2z = elev1 + (self.PartNormDepth * tdepth1)
        self.setz(p2z)
        return elev1, wse1

    def update_info(self, time, bedelev, wse):
        """Update particle information."""
        self.bedElev = bedelev
        self.wse = wse
        self.htabvbed = self.z - self.bedElev
        self.time = time

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

    def interpolate_fields(self, count_index):
        """[Summary]."""
        # Find current location in 2D grid
        point2d = np.vstack((self.x, self.y, np.zeros(self.nparts))).T
        for i in range(self.nparts):
            self.cellindex2d[i] = self.River.CellLocator2D.FindCell(point2d[i, :])
        # if np.any(self.cellindex2d < 0):
        #    print("initial cell -1")  # untested
        # Interpolate 2D fields
        for i in range(self.nparts):
            weights, idlist1, numpts = self.get_cell_pos(
                point2d[i, :], self.cellindex2d[i]
            )
            self.bedElev[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.Elevation_2D
            )
            self.wse[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.WSE_2D
            )
            self.depth[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.Depth_2D
            )
            # self.tmpibc[i] = self.get_cell_value(
            #     weights, idlist1, numpts, self.River.IBC_2D
            # )
            # tmpvel[i] = get_cell_value(weights, idlist1, numpts, Velocity_2D)
            self.shearstress[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.ShearStress2D
            )
            if self.track2d:
                self.velx[i], self.vely[i] = self.get_vel2d_value(
                    weights, idlist1, numpts
                )
        # check shear stress (without error print statements)
        self.shearstress = np.where(self.shearstress < 0.0, 0.0, self.shearstress)
        self.ustar = (self.shearstress / 1000.0) ** 0.5

        # MOVE ELSEWHERE; Check particle depths and calc PartNormDepth
        if count_index <= 1:
            self.check_z(0.5, self.bedElev, self.wse)
        if self.track3d:
            self.PartNormDepth = (self.z - self.bedElev) / self.depth

        # Get 3D Velocity Components
        if self.track3d:
            self.interpolate_field_3d()

    def interpolate_field_3d(self):
        """Locate particle in 3D grid and interpolate velocity."""
        idlist1 = vtk.vtkIdList()
        point3d = np.vstack((self.x, self.y, self.z)).T
        for i in range(self.nparts):
            self.cellindex3d[i] = self.River.CellLocator3D.FindCell(point3d[i, :])
            if self.cellindex3d[i] >= 0:
                (
                    result,
                    dist,
                    tmp3dux,
                    tmp3duy,
                    tmp3duz,
                ) = self.get_vel3d_value(point3d[i, :], self.cellindex3d[i])
                self.velx[i] = tmp3dux
                self.vely[i] = tmp3duy
                self.velz[i] = tmp3duz
            else:
                print("3d findcell failed, particle number: ", i)
                print("switching to FindCellsAlongLine() method")
                pp1 = [point3d[i, 0], point3d[i, 1], self.wse[i] + 10]
                pp2 = [point3d[i, 0], point3d[i, 1], self.bedElev[i] - 10]
                self.River.CellLocator3D.FindCellsAlongLine(pp1, pp2, 0.0, idlist1)
                maxdist = 1e6
                for t in range(idlist1.GetNumberOfIds()):
                    (
                        result,
                        dist,
                        tmp3dux,
                        tmp3duy,
                        tmp3duz,
                    ) = self.get_vel3d_value(point3d[i, :], idlist1.GetId(t))
                    if result == 1:
                        self.velx[i] = tmp3dux
                        self.vely[i] = tmp3duy
                        self.velz[i] = tmp3duz
                        self.cellindex3d[i] = idlist1.GetId(t)
                        break
                    elif dist < maxdist:
                        maxdist = dist
                        self.velx[i] = tmp3dux
                        self.vely[i] = tmp3duy
                        self.velz[i] = tmp3duz
                        self.cellindex3d[i] = idlist1.GetId(t)
                if self.cellindex3d[i] < 0:
                    print("part still out of 3d grid")
                    self.velx[i] = 0.0
                    self.vely[i] = 0.0
                    self.velz[i] = 0.0
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

    def calc_dispersion_coefs(self, lev, bx, by, bz):
        """[summary].

        Args:
            lev ([type]): [description]
            bx ([type]): [description]
            by ([type]): [description]
            bz ([type]): [description]
        """
        ustarh = (self.wse - self.bedElev) * self.ustar
        self.Dx = lev + bx * ustarh
        self.Dy = lev + by * ustarh
        self.Dz = lev + bz * ustarh

    def is_cell_wet_kernel(self, weights, idlist1, numpts):
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
        # cellidb = np.zeros(self.nparts, dtype=np.int64)
        wet = np.empty(self.nparts, dtype=bool)
        for i in range(self.nparts):
            self.cellindex2d[i] = self.River.CellLocator2D.FindCell(newpoint2d[i, :])
            # Create check here on cellindex2d -- particles that cross the downstream river boundary
            #     will trigger cellindex2d[i] < 0 here first; remove particle from list???
            weights, idlist1, numpts = self.get_cell_pos(
                newpoint2d[i, :], self.cellindex2d[i]
            )
            wet[i] = self.is_cell_wet_kernel(weights, idlist1, numpts)
        return wet

    def handle_dry_parts(self, wet1, px, py, dt):
        """[summary]."""
        # print("dry cell encountered")
        # Forward-project dry cells using just random motion
        px[~wet1] = (
            self.x[~wet1] + self.xrnum[~wet1] * (2.0 * self.Dx[~wet1] * dt) ** 0.5
        )
        py[~wet1] = (
            self.y[~wet1] + self.yrnum[~wet1] * (2.0 * self.Dy[~wet1] * dt) ** 0.5
        )
        # Run is_cell_wet again
        newpoint2d = np.vstack((px, py, np.zeros(self.nparts))).T
        wet2 = np.empty(self.nparts, dtype=bool)
        for i in range(self.nparts):  # Expensive
            self.cellindex2d[i] = self.River.CellLocator2D.FindCell(newpoint2d[i, :])
            weights, idlist1, numpts = self.get_cell_pos(
                newpoint2d[i, :], self.cellindex2d[i]
            )
            wet2[i] = self.is_cell_wet_kernel(weights, idlist1, numpts)
        # Any still dry entries will have zero positional update this step
        px[~wet2] = self.x[~wet2]
        py[~wet2] = self.y[~wet2]
        # Move 2D random only for particles that were not wet first time, yes wet second time
        a = ~wet1 & wet2
        self.move_random_only_2d(a, dt)
        # Ensure that move_all() does nothing for these particles
        self.velx[~wet1] = 0.0
        self.vely[~wet1] = 0.0
        self.velz[~wet1] = 0.0
        self.Dx[~wet1] = 0.0
        self.Dy[~wet1] = 0.0
        self.Dz[~wet1] = 0.0
        newpoint2d = np.vstack((px, py, np.zeros(self.nparts))).T
        for i in range(self.nparts):
            self.cellindex2d[i] = self.River.CellLocator2D.FindCell(newpoint2d[i, :])

    def prevent_mindepth(self, px, py, min_depth):
        """[summary]."""
        newpoint2d = np.vstack((px, py, np.zeros(self.nparts))).T
        elev1 = np.zeros(self.nparts)
        wse1 = np.zeros(self.nparts)
        for i in range(self.nparts):
            weights, idlist1, numpts = self.get_cell_pos(
                newpoint2d[i, :], self.cellindex2d[i]
            )
            elev1[i] = self.get_cell_value(
                weights, idlist1, numpts, self.River.Elevation_2D
            )
            wse1[i] = self.get_cell_value(weights, idlist1, numpts, self.River.WSE_2D)
        tdepth1 = wse1 - elev1
        a = tdepth1 < min_depth
        if np.any(a):
            print("particle entered min_depth")
            px[a] = self.x[a]
            py[a] = self.y[a]
            self.velx[a] = 0.0
            self.vely[a] = 0.0
            self.velz[a] = 0.0
            self.Dx[a] = 0.0
            self.Dy[a] = 0.0
            self.Dz[a] = 0.0
            # Eighth, update vertical position to same fractional depth as last
            newpoint2d = np.vstack((px, py, np.zeros(self.nparts))).T
            for i in range(self.nparts):
                self.cellindex2d[i] = self.River.CellLocator2D.FindCell(
                    newpoint2d[i, :]
                )
                weights, idlist1, numpts = self.get_cell_pos(
                    newpoint2d[i, :], self.cellindex2d[i]
                )
                elev1[i] = self.get_cell_value(
                    weights, idlist1, numpts, self.River.Elevation_2D
                )
                wse1[i] = self.get_cell_value(
                    weights, idlist1, numpts, self.River.WSE_2D
                )
        return elev1, wse1

    def gen_rands(self):
        """[summary]."""
        self.xrnum = self.rng.standard_normal(self.nparts)
        self.yrnum = self.rng.standard_normal(self.nparts)
        if self.track3d:
            self.zrnum = self.rng.standard_normal(self.nparts)
