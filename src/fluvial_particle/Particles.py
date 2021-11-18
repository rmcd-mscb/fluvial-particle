"""Particles Class module."""
import h5py
import numpy as np
import vtk


class Particles:
    """A class of particles, each with a velocity, size, and mass."""

    def __init__(self, nparts, x, y, z, rng, mesh, track3d=1):
        """Initialize instance of class Particles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            track3d (bool): 1 if 3D model run, 0 if 2D model run
        """
        self.nparts = nparts
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.rng = rng
        self.mesh = mesh
        self.track3d = track3d

        self._bedelev = np.zeros(nparts)
        self._wse = np.zeros(nparts)
        self._normdepth = np.full(nparts, 0.5)
        self.indices = np.arange(nparts)
        self._cellindex2d = np.full(nparts, fill_value=-1, dtype=np.int64)
        self._cellindex3d = np.full(nparts, fill_value=-1, dtype=np.int64)
        self._depth = np.zeros(nparts)
        self._velx = np.zeros(nparts)
        self._vely = np.zeros(nparts)
        self._velz = np.zeros(nparts)
        self._shearstress = np.zeros(nparts)
        self._ustar = np.zeros(nparts)
        self._diffx = np.zeros(nparts)
        self._diffy = np.zeros(nparts)
        self._diffz = np.zeros(nparts)
        self._time = np.zeros(nparts)
        self._htabvbed = np.zeros(nparts)
        self._mask = None
        self.xrnum = np.zeros(nparts)
        self.yrnum = np.zeros(nparts)
        self.zrnum = np.zeros(nparts)

    def adjust_z(self, pz, alpha):
        """Check that new particle vertical position is within bounds.

        Args:
            pz (float NumPy array): new elevation array
            alpha (float): bounds particle in fractional water column to [alpha, 1-alpha]
        """
        # check on alpha? only makes sense for alpha<=0.5
        a = self.indices[pz > self.wse - alpha * self.depth]
        b = self.indices[pz < self.bedelev + alpha * self.depth]
        pz[a] = self.wse[a] - alpha * self.depth[a]
        pz[b] = self.bedelev[b] + alpha * self.depth[b]

    def calc_diffusion_coefs(self, lev, bx, by, bz):
        """Calculate diffusion coefficients, McDonald & Nelson (2021).

        Args:
            lev ([type]): lateral eddy viscosity
            bx ([type]): coefficient scales x diffusion
            by ([type]): coefficient scales y diffusion
            bz ([type]): coefficient scales z diffusion
        """
        ustarh = self.depth * self.ustar
        self.diffx = lev + bx * ustarh
        self.diffy = lev + by * ustarh
        self.diffz = bz * ustarh  # lev + bz * ustarh

    def create_hdf(self, nprints, globalnparts, comm=None, fname="particles.h5"):
        """Create an HDF5 file to write incremental particles results.

        Args:
            nprints (int): size of first dimension, indexes printing time slices
            globalnparts (int): global number of particles, distributed across processors
            comm (MPI communicator): only for parallel runs
            fname (string): name of the HDF5 file

        Returns:
            parts_h5: new open HDF5 file object
        """
        if comm is None:
            parts_h5 = h5py.File(fname, "w")  # Serial version
        else:
            parts_h5 = h5py.File(fname, "w", driver="mpio", comm=comm)  # MPI version

        grpc = parts_h5.create_group("coordinates")
        grpc.attrs["Description"] = "Position x,y,z of particles at printing time steps"
        grpc.create_dataset("x", (nprints, globalnparts), dtype="f", fillvalue=np.nan)
        grpc.create_dataset("y", (nprints, globalnparts), dtype="f", fillvalue=np.nan)
        grpc.create_dataset("z", (nprints, globalnparts), dtype="f", fillvalue=np.nan)
        grpc.create_dataset("time", (nprints, 1), dtype="f", fillvalue=np.nan)
        grpc["x"].attrs["Units"] = "meters"
        grpc["y"].attrs["Units"] = "meters"
        grpc["z"].attrs["Units"] = "meters"
        grpc["time"].attrs["Units"] = "seconds"

        grpp = parts_h5.create_group("properties")
        grpp.create_dataset(
            "bedelev", (nprints, globalnparts), dtype="f", fillvalue=np.nan
        )
        grpp.create_dataset(
            "cellidx2d", (nprints, globalnparts), dtype="i", fillvalue=-1
        )
        grpp.create_dataset(
            "cellidx3d", (nprints, globalnparts), dtype="i", fillvalue=-1
        )
        grpp.create_dataset(
            "htabvbed", (nprints, globalnparts), dtype="f", fillvalue=np.nan
        )
        grpp.create_dataset(
            "velvec", (nprints, globalnparts, 3), dtype="f", fillvalue=np.nan
        )
        grpp.create_dataset("wse", (nprints, globalnparts), dtype="f", fillvalue=np.nan)
        grpp["bedelev"].attrs[
            "Description"
        ] = "Bed elevation at x,y position of particles"
        grpp["cellidx2d"].attrs[
            "Description"
        ] = "Index of 2D grid cell containing each particle"
        grpp["cellidx3d"].attrs[
            "Description"
        ] = "Index of 3D grid cell containing each particle"
        grpp["htabvbed"].attrs["Description"] = "Height of particle above bed elevation"
        grpp["velvec"].attrs["Description"] = "Velocity vector (u,v,w) of particles"
        grpp["wse"].attrs[
            "Description"
        ] = "Water surface elevation at x,y position of particles"
        grpp["bedelev"].attrs["Units"] = "meters"
        grpp["cellidx2d"].attrs["Units"] = "None"
        grpp["cellidx3d"].attrs["Units"] = "None"
        grpp["htabvbed"].attrs["Units"] = "meters"
        grpp["velvec"].attrs["Units"] = "meters per second"
        grpp["wse"].attrs["Units"] = "meters"
        return parts_h5

    def deactivate_particle(self, idx):
        """Turn off particles that have left the river domain.

        Args:
            idx (int): index of particle to turn off
        """
        self._x[idx] = np.nan
        self._y[idx] = np.nan
        self._z[idx] = np.nan
        self._bedelev[idx] = np.nan
        self._wse[idx] = np.nan
        self._normdepth[idx] = np.nan
        self._cellindex2d[idx] = -1  # integer dtypes cant be nan
        self._cellindex3d[idx] = -1
        self._velx[idx] = np.nan
        self._vely[idx] = np.nan
        self._velz[idx] = np.nan
        self._htabvbed[idx] = np.nan
        self._shearstress[idx] = np.nan
        self._ustar[idx] = np.nan
        self._diffx[idx] = np.nan
        self._diffy[idx] = np.nan
        self._diffz[idx] = np.nan
        self._time[idx] = np.nan

    def gen_rands(self):
        """Generate random numbers drawn from standard normal distribution."""
        self.xrnum = self.rng.standard_normal(self.nparts)
        self.yrnum = self.rng.standard_normal(self.nparts)
        if self.track3d:
            self.zrnum = self.rng.standard_normal(self.nparts)

    def get_total_position(self):
        """Return complete position of particle."""
        return (
            self.time,
            self.cellindex2d,
            self.x,
            self.y,
            self.z,
            self.bedelev,
            self.htabvbed,
            self.wse,
        )

    def handle_dry_parts(self, px, py, a, dt):
        """Adjust trajectories of dry particles.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            a (int NumPy array): indices of dry particles
            dt (float): time step
        """
        # Move dry particles with only 2d random motion
        self.perturb_random_only_2d(px, py, a, dt)
        # Run is_part_wet again
        wet = self.is_part_wet(px, py, a)
        if np.any(~wet):
            b = a[~wet]
            # Any still dry particles will have no positional update this step
            px[b] = self.x[b]
            py[b] = self.y[b]
            # update cell indices
            cell = vtk.vtkGenericCell()
            pcoords = [0.0, 0.0, 0.0]
            weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for i in np.nditer(b, ["zerosize_ok"]):
                point = [px[i], py[i], 0.0]
                cell.SetCellTypeToEmptyCell()
                self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(
                    point,
                    0.0,
                    cell,
                    pcoords,
                    weights,
                )

    def initialize_location(self, frac):
        """Initialize position in water column and interpolate mesh arrays.

        Args:
            frac (float): starting position of particles within water column (scalar or NumPy array)
        """
        # ASSERT check that x,y are within the mesh domain
        # ASSERT check that frac in (epsilon, 1-epsilon)
        self.interp_fields()
        self.z = self.bedelev + frac * self.depth
        self.htabvbed = self.z - self.bedelev
        self.time.fill(0.0)

    def interp_cell_value(self, weights, idlist, numpts, valarray):
        """Interpolate valarray at a point.

        Args:
            weights (float): parametric interpolation weights
            idlist (vtkIdList): list of points that define cell # cellid
            numpts (vtkIdType): number of points in idlist
            valarray (vtk dataset array): field array defined at grid nodes

        Returns:
            interpval (float): interpolated value
        """
        interpval = np.float64(0.0)
        for i in range(numpts):
            interpval += weights[i] * valarray.GetTuple(idlist.GetId(i))[0]

        return interpval

    def interp_field_3d(self):
        """Interpolate 3D velocity field at current particle positions."""
        cell = vtk.vtkGenericCell()
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a = np.copy(self.indices)
        if self.mask is not None:
            a = np.copy(self.indices[self.mask])
        for i in np.nditer(a, ["zerosize_ok"]):
            point = [self.x[i], self.y[i], self.z[i]]
            cell.SetCellTypeToEmptyCell()
            self.cellindex3d[i] = self.mesh.CellLocator3D.FindCell(
                point, 0.0, cell, pcoords, weights
            )
            if self.cellindex3d[i] >= 0:
                idlist = cell.GetPointIds()
                ux, uy, uz = self.interp_vel3d_value(idlist, weights)
                self.velx[i] = ux
                self.vely[i] = uy
                self.velz[i] = uz
            else:
                """print(
                    f"3d findcell failed particle number: {i}, switching to FindCellsAlongLine()"
                )"""
                idlist = vtk.vtkIdList()
                pp1 = [point[0], point[1], self.wse[i] + 10]
                pp2 = [point[0], point[1], self.bedelev[i] - 10]
                self.mesh.CellLocator3D.FindCellsAlongLine(pp1, pp2, 0.0, idlist)
                maxdist = 1e6
                for t in range(idlist.GetNumberOfIds()):
                    (
                        result,
                        dist,
                        tmp3dux,
                        tmp3duy,
                        tmp3duz,
                    ) = self.interp_vel3d_value_alongline(point, idlist.GetId(t))
                    if result == 1:
                        self.velx[i] = tmp3dux
                        self.vely[i] = tmp3duy
                        self.velz[i] = tmp3duz
                        self.cellindex3d[i] = idlist.GetId(t)
                        break
                    elif dist < maxdist:
                        maxdist = dist
                        self.velx[i] = tmp3dux
                        self.vely[i] = tmp3duy
                        self.velz[i] = tmp3duz
                        self.cellindex3d[i] = idlist.GetId(t)
                if self.cellindex3d[i] < 0:
                    print("part still out of 3d grid")
                    self.velx[i] = 0.0
                    self.vely[i] = 0.0
                    self.velz[i] = 0.0
                """print("3d findcell failed, particle number: ", i)
                print("Particle location: ", point3d_2[i, :])
                print("Particle fractional depth: ", normdepth[i])
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

    def interp_fields(self):
        """Interpolate mesh fields at current particle positions."""
        # Find current location in 2D grid and interpolate 2D fields
        cell = vtk.vtkGenericCell()
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a = np.copy(self.indices)
        if self.mask is not None:
            a = np.copy(self.indices[self.mask])
        for i in np.nditer(a, ["zerosize_ok"]):
            point = [self.x[i], self.y[i], 0.0]
            cell.SetCellTypeToEmptyCell()
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(
                point, 0.0, cell, pcoords, weights
            )
            idlist = cell.GetPointIds()
            numpts = cell.GetNumberOfPoints()
            self.bedelev[i] = self.interp_cell_value(
                weights, idlist, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist, numpts, self.mesh.WSE_2D
            )
            # self.depth[i] = self.interp_cell_value(
            #     weights, idlist, numpts, self.mesh.Depth_2D
            # )
            # self.tmpibc[i] = self.interp_cell_value(
            #     weights, idlist, numpts, self.mesh.IBC_2D
            # )
            # tmpvel[i] = interp_cell_value(weights, idlist, numpts, Velocity_2D)
            self.shearstress[i] = self.interp_cell_value(
                weights, idlist, numpts, self.mesh.ShearStress2D
            )
            if ~self.track3d:
                self.velx[i], self.vely[i] = self.interp_vel2d_value(
                    weights, idlist, numpts
                )
        self.depth = self.wse - self.bedelev
        # check shear stress (without error print statements)
        self.shearstress = np.where(self.shearstress < 0.0, 0.0, self.shearstress)
        self.ustar = (self.shearstress / 1000.0) ** 0.5

        if self.track3d:
            self.normdepth = (self.z - self.bedelev) / self.depth
            # Get 3D Velocity Components
            self.interp_field_3d()

    def interp_vel2d_value(self, weights, idlist, numpts):
        """Interpolate 2D velocity vector at a point.

        Args:
            weights (float): parametric interpolation weights
            idlist (vtkIdList): list of points that define cell # cellid
            numpts (vtkIdType): number of points in idlist

        Returns:
            interpxval (float): interpolated x component of vector
            interpyval (float): interpolated y component of vector
        """
        interpxval = np.float64(0.0)
        interpyval = np.float64(0.0)
        for i in range(numpts):
            interpxval += (
                weights[i] * self.mesh.VelocityVec2D.GetTuple(idlist.GetId(i))[0]
            )
            interpyval += (
                weights[i] * self.mesh.VelocityVec2D.GetTuple(idlist.GetId(i))[1]
            )
        return interpxval, interpyval

    def interp_vel3d_value(self, idlist, weights):
        """Interpolate 3D velocity vector at a point contained in a cell.

        Args:
            idlist (vtkIdList): list of points that define the cell
            weights (list): list of interpolation weights

        Returns:
            float: 3D velocity vector components
        """
        interpvalx = np.float64(0.0)
        interpvaly = np.float64(0.0)
        interpvalz = np.float64(0.0)

        for i in range(8):
            a = self.mesh.VelocityVec3D.GetTuple(idlist.GetId(i))
            interpvalx += weights[i] * a[0]
            interpvaly += weights[i] * a[1]
            interpvalz += weights[i] * a[2]
        return interpvalx, interpvaly, interpvalz

    def interp_vel3d_value_alongline(self, point, cellid):
        """Interpolate 3D velocity vector at a point.

        Args:
            point (float): position vector (x,y,z) of the particle
            cellid (int): index of the 2D grid cell containing the particle

        Returns:
            [type]: [description]
        """
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        clspoint = [0.0, 0.0, 0.0]
        tmpid = vtk.mutable(0)
        vtkid = vtk.mutable(0)
        vtkcell3d = self.mesh.vtksgrid3d.GetCell(cellid)
        result = vtkcell3d.EvaluatePosition(
            point, clspoint, tmpid, pcoords, vtkid, weights
        )
        numpts = vtkcell3d.GetNumberOfPoints()
        idlist = vtkcell3d.GetPointIds()
        interpxval = np.float64(0.0)
        interpyval = np.float64(0.0)
        interpzval = np.float64(0.0)
        for i in range(numpts):
            a = self.mesh.VelocityVec3D.GetTuple(idlist.GetId(i))
            interpxval += weights[i] * a[0]
            interpyval += weights[i] * a[1]
            interpzval += weights[i] * a[2]
        return result, vtkid, interpxval, interpyval, interpzval

    def is_part_wet(self, px, py, a):
        """Determine if particle's new positions are wet.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            a (int NumPy array): indices of particles to check

        Returns:
            wet (boolean NumPy array): True indices mean wet, False means dry
        """
        cell = vtk.vtkGenericCell()
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        wet = np.empty(np.size(a), dtype=bool)
        j = 0
        for i in np.nditer(a, ["zerosize_ok"]):
            point = [px[i], py[i], 0.0]
            cell.SetCellTypeToEmptyCell()
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(
                point, 0.0, cell, pcoords, weights
            )
            if self.cellindex2d[i] < 0:
                if self.mask is None:
                    self.mask = np.full(self.nparts, fill_value=True)
                self.mask[i] = False
                self.deactivate_particle(i)
                wet[j] = False
            else:
                idlist = cell.GetPointIds()
                numpts = cell.GetNumberOfPoints()
                wet[j] = self.is_part_wet_kernel(i, idlist, numpts)
            j += 1
        return wet

    def is_part_wet_kernel(self, idx, idlist, numpts):
        """Interpolate IBC mesh array to a point.

        Args:
            idx ([type]): [description]
            idlist (vtkIdList): list of points that define cell # cellid
            numpts (vtkIdType): number of points in idlist

        Returns:
            (bool): returns True if all points defining cell are wet, False otherwise
        """
        a = np.zeros((numpts,), dtype=np.int32)
        for i in range(numpts):
            a[i] = idlist.GetId(i)
        for i in range(numpts):
            b = self.mesh.IBC_2D.GetTuple(a[i])[0]
            if b < 1:
                return False
        return True

    def move_all(self, alpha, min_depth, time, dt):
        """Update particle positions.

        Args:
            alpha (float): bounding scalar for adjust_z
            min_depth (float): minimum depth scalar that particles can enter
            time (float): the new time at end of position update
            dt (float): time step
        """
        # first perturb 2d only
        px = np.copy(self.x)
        py = np.copy(self.y)
        self.perturb_2d(px, py, dt)

        # check if new positions are wet
        a = np.copy(self.indices)
        if self.mask is not None:
            a = a[self.mask]
        wet = self.is_part_wet(px, py, a)
        if np.any(~wet):
            self.handle_dry_parts(px, py, a[~wet], dt)
        if self.mask is not None:
            a = np.copy(self.indices[self.mask])

        # update bed elevation, wse, depth
        cell = vtk.vtkGenericCell()
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in np.nditer(a, ["zerosize_ok"]):
            point = [px[i], py[i], 0.0]
            cell.SetCellTypeToEmptyCell()
            self.mesh.CellLocator2D.FindCell(point, 0.0, cell, pcoords, weights)
            idlist = cell.GetPointIds()
            numpts = cell.GetNumberOfPoints()
            self.bedelev[i] = self.interp_cell_value(
                weights, idlist, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist, numpts, self.mesh.WSE_2D
            )
        self.depth = self.wse - self.bedelev

        # Prevent particles from entering cells where depth < min_depth
        if np.any(self.depth < min_depth):
            self.prevent_mindepth(px, py, min_depth)

        # Perturb vertical, random wiggle by default
        # subclasses can update perturb_z for an active drift
        pz = self.perturb_z(dt)
        self.adjust_z(pz, alpha)

        # Move particles
        self.x = px
        self.y = py
        self.z = pz

        # Update some info
        self.htabvbed = self.z - self.bedelev
        self.time.fill(time)

    def perturb_2d(self, px, py, dt):
        """Project particles' 2D trajectories based on starting interpolated quantities.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            dt (float): time step
        """
        vx = self.velx
        vy = self.vely
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = self.xrnum * (2.0 * self.diffx * dt) ** 0.5
        yranwalk = self.yrnum * (2.0 * self.diffy * dt) ** 0.5
        # Move and update positions in-place on each array
        a = self.indices[velmag > 0.0]
        b = self.indices[velmag == 0.0]
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
        px[b] += xranwalk[b]
        py[b] += yranwalk[b]

    def perturb_random_only_2d(self, px, py, a, dt):
        """Project new particle 2D positions based on random walk only.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            a (int NumPy array): indices of dry particles
            dt (float): time step
        """
        px[a] = self.x[a] + self.xrnum[a] * (2.0 * self.diffx[a] * dt) ** 0.5
        py[a] = self.y[a] + self.yrnum[a] * (2.0 * self.diffy[a] * dt) ** 0.5

    def perturb_z(self, dt):
        """Project particles' vertical trajectories, random wiggle.

        Args:
            dt (float): time step

        Returns:
            pz (float NumPy array): new elevation array
        """
        zranwalk = self.zrnum * (2.0 * self.diffz * dt) ** 0.5
        pz = self.bedelev + (self.normdepth * self.depth) + self.velz * dt + zranwalk
        return pz

    def prevent_mindepth(self, px, py, min_depth):
        """Prevent particles from entering a position with depth < min_depth.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            min_depth (float): minimum allowed depth that particles may enter
        """
        # print("particle entered min_depth")
        cell = vtk.vtkGenericCell()
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a = np.copy(self.indices[self.depth < min_depth])
        # update cell indices and interpolations
        for i in np.nditer(a, ["zerosize_ok"]):
            px[i] = self.x[i]
            py[i] = self.y[i]
            point = [px[i], py[i], 0.0]
            cell.SetCellTypeToEmptyCell()
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(
                point, 0.0, cell, pcoords, weights
            )
            idlist = cell.GetPointIds()
            numpts = cell.GetNumberOfPoints()
            self.bedelev[i] = self.interp_cell_value(
                weights, idlist, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist, numpts, self.mesh.WSE_2D
            )
        self.depth[a] = self.wse[a] - self.bedelev[a]

    def write_hdf5(self, obj, tidx, start, end, time, rank):
        """Write particle positions and interpolated quantities to file.

        Args:
            obj (h5py file): open h5py file object created with self.create_hdf()
            tidx (int): time slice index
            start (int): starting index of this processors assigned write space
            end (int): ending index (non-inclusive)
            time (float): current simulation time
            rank (int): processor rank if run in MPI (0 in serial)
        """
        grpc = obj["coordinates"]
        grpp = obj["properties"]
        grpc["x"][tidx, start:end] = self.x
        grpc["y"][tidx, start:end] = self.y
        grpc["z"][tidx, start:end] = self.z
        if rank == 0:
            grpc["time"][tidx] = time
        grpp["bedelev"][tidx, start:end] = self.bedelev
        grpp["htabvbed"][tidx, start:end] = self.htabvbed
        grpp["wse"][tidx, start:end] = self.wse
        grpp["velvec"][tidx, start:end, :] = np.vstack(
            (self.velx, self.vely, self.velz)
        ).T
        grpp["cellidx2d"][tidx, start:end] = self.cellindex2d
        grpp["cellidx3d"][tidx, start:end] = self.cellindex3d

    def write_hdf5_xmf(self, filexmf, time, nprints, nparts, tidx):
        """Write the particles xmf file for visualizations in Paraview.

        Args:
            filexmf (file): open file to write
            time (float): current simulation time
            nprints (int): total number of printing steps
            nparts (int): global number of particles summed across processors
            tidx (int): time slice index corresponding to time
        """
        filexmf.write(
            f"""
            <Grid GridType="Uniform">
                <Time Value="{time}"/>
                <Topology NodesPerElement="{nparts}" TopologyType="Polyvertex"/>
                <Geometry GeometryType="X_Y_Z" Name="particles">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                            {tidx} 0
                            1 1
                            1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF">
                            particles.h5:/coordinates/x
                        </DataItem>
                    </DataItem>
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                            {tidx} 0
                            1 1
                            1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF">
                            particles.h5:/coordinates/y
                        </DataItem>
                    </DataItem>
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                            {tidx} 0
                            1 1
                            1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF">
                            particles.h5:/coordinates/z
                        </DataItem>
                    </DataItem>
                </Geometry>
                <Attribute Name="BedElevation" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF">
                            particles.h5:/properties/bedelev
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="CellIndex2D" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF">
                            particles.h5:/properties/cellidx2d
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="CellIndex3D" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF">
                            particles.h5:/properties/cellidx3d
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="HeightAboveBed" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF">
                            particles.h5:/properties/htabvbed
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="WaterSurfaceElevation" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF">
                            particles.h5:/properties/wse
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="VelocityVector" AttributeType="Vector" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts} 3" Format="XML">
                        <DataItem Dimensions="3 3" Format="XML">
                        {tidx} 0 0
                        1 1 1
                        1 {nparts} 3
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts} 3" Format="HDF">
                            particles.h5:/properties/velvec
                        </DataItem>
                    </DataItem>
                </Attribute>
            </Grid>"""
        )

    def write_hdf5_xmf_footer(self, filexmf):
        """Write final lines of XDMF file.

        Args:
            filexmf (file): open file to write
        """
        filexmf.write(
            """
                </Grid>
            </Domain>
        </Xdmf>
        """
        )

    def write_hdf5_xmf_header(self, filexmf):
        """Write initial lines of XDMF file.

        Args:
            filexmf (file): open file to write
        """
        filexmf.write(
            """<Xdmf Version="3.0">
            <Domain>
                <Grid GridType="Collection" CollectionType="Temporal">"""
        )

    # Properties

    @property
    def bedelev(self):
        """Get bedelev.

        Returns:
            [type]: [description]
        """
        return self._bedelev

    @bedelev.setter
    def bedelev(self, values):
        """Set bedelev.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "bedelev.setter wrong size"
        )
        self._bedelev = values

    @property
    def cellindex2d(self):
        """Get cellindex2d.

        Returns:
            [type]: [description]
        """
        return self._cellindex2d

    @cellindex2d.setter
    def cellindex2d(self, values):
        """Set cellindex2d.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "cellindex2d.setter wrong size"
        )
        self._cellindex2d = values

    @property
    def cellindex3d(self):
        """Get cellindex3d.

        Returns:
            [type]: [description]
        """
        return self._cellindex3d

    @cellindex3d.setter
    def cellindex3d(self, values):
        """Set cellindex3d.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "cellindex3d.setter wrong size"
        )
        self._cellindex3d = values

    @property
    def depth(self):
        """Get depth.

        Returns:
            [type]: [description]
        """
        return self._depth

    @depth.setter
    def depth(self, values):
        """Set depth.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "depth.setter wrong size"
        )
        self._depth = values

    @property
    def diffx(self):
        """Get diffx.

        Returns:
            [type]: [description]
        """
        return self._diffx

    @diffx.setter
    def diffx(self, values):
        """Set diffx.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "diffx.setter wrong size"
        )
        self._diffx = values

    @property
    def diffy(self):
        """Get diffy.

        Returns:
            [type]: [description]
        """
        return self._diffy

    @diffy.setter
    def diffy(self, values):
        """Set diffy.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "diffy.setter wrong size"
        )
        self._diffy = values

    @property
    def diffz(self):
        """Get diffz.

        Returns:
            [type]: [description]
        """
        return self._diffz

    @diffz.setter
    def diffz(self, values):
        """Set diffz.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "diffz.setter wrong size"
        )
        self._diffz = values

    @property
    def htabvbed(self):
        """Get htabvbed.

        Returns:
            [type]: [description]
        """
        return self._htabvbed

    @htabvbed.setter
    def htabvbed(self, values):
        """Set htabvbed.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "htabvbed.setter wrong size"
        )
        self._htabvbed = values

    @property
    def mask(self):
        """Get mask.

        Returns:
            [type]: [description]
        """
        return self._mask

    @mask.setter
    def mask(self, values):
        """Set mask.

        Args:
            values ([type]): [description]
        """
        assert np.issubdtype(values.dtype, "bool"), ValueError(  # noqa: S101
            "mask.setter: mask must be of 'bool' data type"
        )
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "mask.setter wrong size"
        )
        self._mask = values

    @property
    def mesh(self):
        """Get mesh.

        Returns:
            [type]: [description]
        """
        return self._mesh

    @mesh.setter
    def mesh(self, values):
        """Set mesh.

        Args:
            values ([type]): [description]
        """
        self._mesh = values

    @property
    def normdepth(self):
        """Get normdepth.

        Returns:
            [type]: [description]
        """
        return self._normdepth

    @normdepth.setter
    def normdepth(self, values):
        """Set normdepth.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "normdepth.setter wrong size"
        )
        self._normdepth = values

    @property
    def nparts(self):
        """Get nparts.

        Returns:
            [type]: [description]
        """
        return self._nparts

    @nparts.setter
    def nparts(self, value):
        """Set nparts.

        Args:
            value ([type]): [description]
        """
        assert value > 0, ValueError("# particles < 1")  # noqa: S101
        assert isinstance(value, int), TypeError("nparts must be int")  # noqa: S101
        self._nparts = value

    @property
    def rng(self):
        """Get rng.

        Returns:
            [type]: [description]
        """
        return self._rng

    @rng.setter
    def rng(self, values):
        """Set rng.

        Args:
            values ([type]): [description]
        """
        self._rng = values

    @property
    def shearstress(self):
        """Get shearstress.

        Returns:
            [type]: [description]
        """
        return self._shearstress

    @shearstress.setter
    def shearstress(self, values):
        """Set shearstress.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "shearstress.setter wrong size"
        )
        self._shearstress = values

    @property
    def time(self):
        """Get time.

        Returns:
            [type]: [description]
        """
        return self._time

    @time.setter
    def time(self, values):
        """Set time.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "time.setter wrong size"
        )
        self._time = values

    @property
    def track3d(self):
        """Get track3d.

        Returns:
            [type]: [description]
        """
        return self._track3d

    @track3d.setter
    def track3d(self, value):
        """Set track3d.

        Args:
            value ([type]): [description]
        """
        assert isinstance(value, int), TypeError("track3d must be int")  # noqa: S101
        assert value >= 0 and value < 2, ValueError(  # noqa: S101
            "track3d must be 0 or 1"
        )
        self._track3d = value

    @property
    def ustar(self):
        """Get ustar.

        Returns:
            [type]: [description]
        """
        return self._ustar

    @ustar.setter
    def ustar(self, values):
        """Set ustar.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "ustar.setter wrong size"
        )
        self._ustar = values

    @property
    def velx(self):
        """Get velx.

        Returns:
            [type]: [description]
        """
        return self._velx

    @velx.setter
    def velx(self, values):
        """Set velx.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "velx.setter wrong size"
        )
        self._velx = values

    @property
    def vely(self):
        """Get vely.

        Returns:
            [type]: [description]
        """
        return self._vely

    @vely.setter
    def vely(self, values):
        """Set vely.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "vely.setter wrong size"
        )
        self._vely = values

    @property
    def velz(self):
        """Get velz.

        Returns:
            [type]: [description]
        """
        return self._velz

    @velz.setter
    def velz(self, values):
        """Set velz.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "velz.setter wrong size"
        )
        self._velz = values

    @property
    def wse(self):
        """Get wse.

        Returns:
            [type]: [description]
        """
        return self._wse

    @wse.setter
    def wse(self, values):
        """Set wse.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "wse.setter wrong size"
        )
        self._wse = values

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
            "x.setter wrong size"
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
            "y.setter wrong size"
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
            "z.setter wrong size"
        )
        self._z = values
