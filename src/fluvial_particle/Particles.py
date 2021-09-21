"""Particles Class module."""
import numpy as np
import vtk


class Particles:
    """A class of particles, each with a velocity, size, and mass."""

    def __init__(self, nparts, x, y, z, rng, mesh, track2d=0, track3d=1):
        """Initialize instance of class Particles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            track2d (bool): 1 if 2D model run, 0 else
            track3d (bool): 1 if 3D model run, 0 else
        """
        self.nparts = nparts
        self._x = np.copy(x)
        self._y = np.copy(y)
        self._z = np.copy(z)
        self.rng = rng
        self.mesh = mesh
        self.track2d = track2d
        self.track3d = track3d
        # Add an XOR on track2d & track3d ?

        self.time = np.zeros(nparts)
        self.bedElev = np.zeros(nparts)
        self.htabvbed = np.zeros(nparts)
        self.wse = np.zeros(nparts)
        self.PartNormDepth = np.full(nparts, 0.5)
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

    def adjust_z(self, pz, alpha):
        """Check that new particle vertical position is within bounds.

        Args:
            pz ([type]): [description]
            alpha (float): bounds particle in fractional water column to [alpha, 1-alpha]
        """
        # check on alpha? only makes sense for alpha<=0.5
        a = pz > self.wse - alpha * self.depth
        b = pz < self.bedElev + alpha * self.depth
        pz[a] = self.wse[a] - alpha * self.depth[a]
        pz[b] = self.bedElev[b] + alpha * self.depth[b]

    def calc_dispersion_coefs(self, lev, bx, by, bz):
        """Calculate dispersion coefficients.

        Args:
            lev ([type]): [description]
            bx ([type]): [description]
            by ([type]): [description]
            bz ([type]): [description]
        """
        ustarh = self.depth * self.ustar
        self.Dx = lev + bx * ustarh
        self.Dy = lev + by * ustarh
        self.Dz = lev + bz * ustarh

    def gen_rands(self):
        """Generate standard normal random numbers."""
        self.xrnum = self.rng.standard_normal(self.nparts)
        self.yrnum = self.rng.standard_normal(self.nparts)
        if self.track3d:
            self.zrnum = self.rng.standard_normal(self.nparts)

    def interp_cell_value(self, weights, idlist1, numpts, valarray):
        """Interpolate valarray at a point.

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

    def get_pos_in_2dcell(self, newpoint2d, cellid):
        """Find position in 2D cell, return info for interpolation.

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
        vtkcell2d = self.mesh.vtksgrid2d.GetCell(cellid)
        tmpres = vtkcell2d.EvaluatePosition(  # noqa F841
            newpoint2d, clspoint, tmpid, pcoords, vtkid2, weights
        )
        numpts = vtkcell2d.GetNumberOfPoints()
        idlist1 = vtkcell2d.GetPointIds()
        return weights, idlist1, numpts

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

    def handle_dry_parts(self, px, py, dry, dt):
        """Adjust trajectories of dry particles.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            dry ([type]): [description]
            dt ([type]): [description]
        """
        # Move dry particles with only 2d random motion
        a = np.arange(self.nparts)
        a = a[dry]
        self.perturb_random_only_2d(px, py, a, dt)
        # Run is_part_wet again
        # wet2 = self.is_part_wet(px, py)
        wet2 = np.empty(np.size(a), dtype=bool)
        j = 0
        for i in np.nditer(a):  # range(self.nparts):
            point = [px[i], py[i], 0.0]
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)
            weights, idlist1, numpts = self.get_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            wet2[j] = self.is_part_wet_kernel(weights, idlist1, numpts)
            j += 1
        if np.any(~wet2):
            b = a[~wet2]
            # Any still dry particles will have no positional update this step
            px[b] = self.x[b]
            py[b] = self.y[b]
            # Ensure that move_all() does nothing for any of these particles
            self.velx[a] = 0.0
            self.vely[a] = 0.0
            self.velz[a] = 0.0
            self.Dx[a] = 0.0
            self.Dy[a] = 0.0
            self.Dz[a] = 0.0
            # update cell indices
            for i in np.nditer(b):  # range(self.nparts):
                point = [px[i], py[i], 0.0]
                self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)

    def initialize_location(self, frac):
        """Initialize position in water column and interpolate mesh arrays.

        Args:
            frac ([type]): [description]
        """
        # ASSERT check that x,y are within the mesh domain
        # ASSERT check that frac in (epsilon, 1-epsilon)
        self.interp_fields
        self.z = self.bedElev + frac * self.depth

    @property
    def interp_field_3d(self):
        """Interpolate 3D velocity field at current particles' positions."""
        idlist1 = vtk.vtkIdList()
        # point3d = np.vstack((self.x, self.y, self.z)).T
        for i in range(self.nparts):
            point = [self.x[i], self.y[i], self.z[i]]
            self.cellindex3d[i] = self.mesh.CellLocator3D.FindCell(point)
            if self.cellindex3d[i] >= 0:
                (
                    result,
                    dist,
                    tmp3dux,
                    tmp3duy,
                    tmp3duz,
                ) = self.interp_vel3d_value(point, self.cellindex3d[i])
                self.velx[i] = tmp3dux
                self.vely[i] = tmp3duy
                self.velz[i] = tmp3duz
            else:
                print("3d findcell failed, particle number: ", i)
                print("switching to FindCellsAlongLine() method")
                pp1 = [point[0], point[1], self.wse[i] + 10]
                pp2 = [point[0], point[1], self.bedElev[i] - 10]
                self.mesh.CellLocator3D.FindCellsAlongLine(pp1, pp2, 0.0, idlist1)
                maxdist = 1e6
                for t in range(idlist1.GetNumberOfIds()):
                    (
                        result,
                        dist,
                        tmp3dux,
                        tmp3duy,
                        tmp3duz,
                    ) = self.interp_vel3d_value(point, idlist1.GetId(t))
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

    @property
    def interp_fields(self):
        """Interpolate mesh fields at current particles' positions."""
        # Find current location in 2D grid
        # point2d = np.vstack((self.x, self.y, np.zeros(self.nparts))).T
        for i in range(self.nparts):
            point = [self.x[i], self.y[i], 0.0]
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)
        # if np.any(self.cellindex2d < 0):
        #    print("initial cell -1")  # untested
        # Interpolate 2D fields
        for i in range(self.nparts):
            point = [self.x[i], self.y[i], 0.0]
            weights, idlist1, numpts = self.get_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            self.bedElev[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.WSE_2D
            )
            # self.depth[i] = self.interp_cell_value(
            #     weights, idlist1, numpts, self.mesh.Depth_2D
            # )
            # self.tmpibc[i] = self.interp_cell_value(
            #     weights, idlist1, numpts, self.mesh.IBC_2D
            # )
            # tmpvel[i] = interp_cell_value(weights, idlist1, numpts, Velocity_2D)
            self.shearstress[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.ShearStress2D
            )
            if self.track2d:
                self.velx[i], self.vely[i] = self.interp_vel2d_value(
                    weights, idlist1, numpts
                )
        self.depth = self.wse - self.bedElev
        # check shear stress (without error print statements)
        self.shearstress = np.where(self.shearstress < 0.0, 0.0, self.shearstress)
        self.ustar = (self.shearstress / 1000.0) ** 0.5

        if self.track3d:
            self.PartNormDepth = (self.z - self.bedElev) / self.depth
            # Get 3D Velocity Components
            self.interp_field_3d

    def interp_vel2d_value(self, weights, idlist1, numpts):
        """Interpolate 2D velocity vector at a point.

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
                weights[i] * self.mesh.VelocityVec2D.GetTuple(idlist1.GetId(i))[0]
            )
            tmpyval += (
                weights[i] * self.mesh.VelocityVec2D.GetTuple(idlist1.GetId(i))[1]
            )
        return tmpxval, tmpyval

    def interp_vel3d_value(self, newpoint3d, cellid):
        """Interpolate 3D velocity vector at a point.

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
        vtkcell3d = self.mesh.vtksgrid3d.GetCell(cellid)
        result = vtkcell3d.EvaluatePosition(
            newpoint3d, clspoint, tmpid, pcoords, vtkid2, weights
        )
        numpts = vtkcell3d.GetNumberOfPoints()
        idlist1 = vtkcell3d.GetPointIds()
        tmpxval = np.float64(0.0)
        tmpyval = np.float64(0.0)
        for i in range(0, numpts):
            tmpxval += (
                weights[i] * self.mesh.VelocityVec3D.GetTuple(idlist1.GetId(i))[0]
            )
            tmpyval += (
                weights[i] * self.mesh.VelocityVec3D.GetTuple(idlist1.GetId(i))[1]
            )
        return result, vtkid2, tmpxval, tmpyval, 0.0

    def is_part_wet_kernel(self, weights, idlist1, numpts):
        """Interpolate IBC mesh array to a point.

        Args:
            weights ([type]): [description]
            idlist1 ([type]): [description]
            numpts ([type]): [description]

        Returns:
            [type]: [description]
        """
        tmpibc = 0.0
        for i in range(numpts):
            tmpibc += weights[i] * self.mesh.IBC_2D.GetTuple(idlist1.GetId(i))[0]
        if tmpibc >= 0.9999999:
            return True
        else:
            return False

    def is_part_wet(self, px, py):
        """Determine if particles' new positions is wet.

        Args:
            px ([type]): [description]
            py ([type]): [description]

        Returns:
            [type]: [description]
        """
        # newpoint2d = np.vstack((self.x, self.y, np.zeros(self.nparts))).T
        wet = np.empty(self.nparts, dtype=bool)
        for i in range(self.nparts):
            point = [px[i], py[i], 0.0]
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)
            weights, idlist1, numpts = self.get_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            wet[i] = self.is_part_wet_kernel(weights, idlist1, numpts)
        return wet

    def move_all(self, alpha, min_depth, dt):
        """Update position based on speed, angle.

        Args:
            dt (float): time step
            min_depth (float): minimum depth scalar that particles can enter
            alpha (float): bounding scalar for adjust_z
        """
        # first move 2d only
        px = np.copy(self.x)
        py = np.copy(self.y)
        self.perturb_2d(px, py, dt)

        # check if new positions are wet
        wet = self.is_part_wet(px, py)
        if np.any(~wet):
            self.handle_dry_parts(px, py, ~wet, dt)

        # update bed elevation, wse, depth
        for i in range(self.nparts):
            point = [px[i], py[i], 0.0]
            weights, idlist1, numpts = self.get_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            self.bedElev[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.WSE_2D
            )
        self.depth = self.wse - self.bedElev

        # Prevent particles from entering cells where tdepth1 < min_depth
        if np.any(self.depth < min_depth):
            self.prevent_mindepth(px, py, min_depth)

        # Update particle elevation in new water column to same fractional depth as last
        zranwalk = self.zrnum * (2.0 * self.Dz * dt) ** 0.5
        pz = (
            self.bedElev + (self.PartNormDepth * self.depth) + self.velz * dt + zranwalk
        )
        self.adjust_z(pz, alpha)
        self.x = px
        self.y = py
        self.z = pz  # setz(pz)

    def perturb_2d(self, px, py, dt):
        """Project particles' 2D trajectories.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            dt ([type]): [description]
        """
        vx = self.velx
        vy = self.vely
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = self.xrnum * (2.0 * self.Dx * dt) ** 0.5
        yranwalk = self.yrnum * (2.0 * self.Dy * dt) ** 0.5
        # Move and update positions in-place on each array
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

    def perturb_random_only_2d(self, px, py, idx, dt):
        """Update position based on random walk in x and y directions.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            idx ([type]): [description]
            dt ([type]): [description]
        """
        px[idx] = self.x[idx] + self.xrnum[idx] * (2.0 * self.Dx[idx] * dt) ** 0.5
        py[idx] = self.y[idx] + self.yrnum[idx] * (2.0 * self.Dy[idx] * dt) ** 0.5

    def prevent_mindepth(self, px, py, min_depth):
        """Prevent particles from entering a position with depth < min_depth.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            min_depth ([type]): [description]
        """
        print("particle entered min_depth")
        a = self.depth < min_depth
        b = np.arange(self.nparts)
        a = b[a]
        px[a] = self.x[a]
        py[a] = self.y[a]
        self.velx[a] = 0.0
        self.vely[a] = 0.0
        self.velz[a] = 0.0
        self.Dx[a] = 0.0
        self.Dy[a] = 0.0
        self.Dz[a] = 0.0
        # update cell indices and interpolations
        # newpoint2d = np.vstack((px, py, np.zeros(self.nparts))).T
        for i in np.nditer(a):  # range(self.nparts):
            point = [px[i], py[i], 0.0]
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)
            weights, idlist1, numpts = self.get_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            self.bedElev[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.WSE_2D
            )
        self.depth[a] = self.wse[a] - self.bedElev[a]

    def update_info(self, time, bedelev, wse):
        """Update particle information.

        Args:
            time ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]
        """
        self.bedElev = bedelev
        self.wse = wse
        self.htabvbed = self.z - self.bedElev
        self.time = time

    @property
    def x(self):
        """Get x.

        Returns:
            [type]: [description]
        """
        return self._x

    @x.setter
    def x(self, values):
        """Set x.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "x.setter wrong size etc. etc."
        )
        self._x = values

    @property
    def y(self):
        """Get y.

        Returns:
            [type]: [description]
        """
        return self._y

    @y.setter
    def y(self, values):
        """Set y.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "y.setter wrong size etc. etc."
        )
        self._y = values

    @property
    def z(self):
        """Get z.

        Returns:
            [type]: [description]
        """
        return self._z

    @z.setter
    def z(self, values):
        """Set z.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "z.setter wrong size etc. etc."
        )
        self._z = values
