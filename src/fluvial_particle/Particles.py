"""Particles Class module."""
import h5py
import numpy as np
import vtk
from vtk.util import numpy_support  # type:ignore


class Particles:
    """A class of particles, each with a velocity, size, and mass."""

    def __init__(self, nparts, x, y, z, rng, mesh, track3d=1, comm=None):
        """Initialize instance of class Particles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            track3d (bool): 1 if 3D model run, 0 if 2D model run
            comm (mpi4py object): MPI communicator used in parallel execution
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

        # Objects for 2d grid interpolation
        self.pt2d_np = np.zeros((self.nparts, 3))  # ordering required by vtk
        self.pt2d_np[:, 0] = self.x
        self.pt2d_np[:, 1] = self.y
        self.pt2d_vtk = numpy_support.numpy_to_vtk(self.pt2d_np)
        self.pt2d = vtk.vtkPoints()
        self.ptset2d = vtk.vtkPointSet()  # vtkPointSet() REQUIRES vtk>=9.1
        if comm is None:
            self.probe2d = vtk.vtkProbeFilter()
        else:
            self.probe2d = vtk.vtkPProbeFilter()  # parallel version
        self.strategy2d = vtk.vtkCellLocatorStrategy()  # requires vtk>=9.0
        self.pt2d.SetData(self.pt2d_vtk)
        self.ptset2d.SetPoints(self.pt2d)
        self.probe2d.SetInputData(self.ptset2d)
        self.probe2d.SetSourceData(self.mesh.vtksgrid2d)
        self.probe2d.SetFindCellStrategy(self.strategy2d)
        # Objects for 3d grid interpolation
        if self.track3d:
            self.pt3d_np = np.zeros((self.nparts, 3))
            self.pt3d_np[:, 0] = self.x
            self.pt3d_np[:, 1] = self.y
            self.pt3d_np[:, 2] = self.z
            self.pt3d_vtk = numpy_support.numpy_to_vtk(self.pt3d_np)
            self.pt3d = vtk.vtkPoints()
            self.pts3d = vtk.vtkPointSet()
            if comm is None:
                self.probe3d = vtk.vtkProbeFilter()
            else:
                self.probe3d = vtk.vtkPProbeFilter()
            strategy3d = vtk.vtkCellLocatorStrategy()
            self.pt3d.SetData(self.pt3d_vtk)
            self.pts3d.SetPoints(self.pt3d)
            self.probe3d.SetInputData(self.pts3d)
            self.probe3d.SetSourceData(self.mesh.vtksgrid3d)
            self.probe3d.SetFindCellStrategy(strategy3d)
            """ these tolerance functions seem to have no effect for points right on the edge of the 3d cells
            self.probe.ComputeToleranceOff()  # disables automated tolerance calculation
            self.probe.SetTolerance(0.01)  # is this a dimensionless tolerance? how defined?"""

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

    def create_hdf5(self, nprints, globalnparts, comm=None, fname="particles.h5"):
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
        grpc.create_dataset(
            "x", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grpc.create_dataset(
            "y", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grpc.create_dataset(
            "z", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grpc.create_dataset("time", (nprints, 1), dtype=np.float64, fillvalue=np.nan)
        grpc["x"].attrs["Units"] = "meters"
        grpc["y"].attrs["Units"] = "meters"
        grpc["z"].attrs["Units"] = "meters"
        grpc["time"].attrs["Units"] = "seconds"

        grpp = parts_h5.create_group("properties")
        grpp.create_dataset(
            "bedelev", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grpp.create_dataset(
            "cellidx2d", (nprints, globalnparts), dtype=np.int64, fillvalue=-1
        )
        grpp.create_dataset(
            "cellidx3d", (nprints, globalnparts), dtype=np.int64, fillvalue=-1
        )
        grpp.create_dataset(
            "htabvbed", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grpp.create_dataset(
            "velvec", (nprints, globalnparts, 3), dtype=np.float64, fillvalue=np.nan
        )
        grpp.create_dataset(
            "wse", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
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
        self.mask[idx] = False

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

    def handle_dry_parts(self, px, py, dt):
        """Adjust trajectories of dry particles.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            dt (float): time step
        """
        idx = self.indices
        if self.mask is not None:
            idx = idx[self.mask]
        wet = self.is_part_wet(px, py, idx)
        if np.any(~wet):
            a = idx[~wet]
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

    def interp_field_3d(self, px=None, py=None, pz=None):
        """Interpolate 3D velocity field at current particle positions."""
        if px is None:
            px = self.x
        if py is None:
            py = self.y
        if pz is None:
            pz = self.z
        if self.mask is None:
            # Update the NumPy position array
            self.pt3d_np[:, 0] = px
            self.pt3d_np[:, 1] = py
            self.pt3d_np[:, 2] = pz
        else:
            idx = self.indices[self.mask]
            if idx.size != self.pt3d.GetNumberOfPoints():
                # Number of active particles has changed, reconstruct the probe pipeline objects
                # assumes particles are only deactivated, never re-activated (at least simultaneously)
                self.pt3d_np = np.zeros((idx.size, 3))
                self.pt3d_vtk = numpy_support.numpy_to_vtk(self.pt3d_np)
                self.pt3d.Reset()
                self.pt3d.SetData(self.pt3d_vtk)
                self.pts3d.Initialize()
                self.pts3d.SetPoints(self.pt3d)
                self.probe3d.SetInputData(self.pts3d)
            self.pt3d_np[:, 0] = px[idx]
            self.pt3d_np[:, 1] = py[idx]
            self.pt3d_np[:, 2] = pz[idx]
        # Tell downstream VTK objects that the input pipeline has been modified
        self.pt3d.Modified()
        self.probe3d.Update()

        # Get interpolated point data from the probe filter
        dataout = self.probe3d.GetOutput().GetPointData().GetArray("Velocity")
        vel = numpy_support.vtk_to_numpy(dataout)
        if self.mask is None:
            self.velx = vel[:, 0]
            self.vely = vel[:, 1]
            self.velz = vel[:, 2]
        else:
            self.velx[idx] = vel[:, 0]
            self.vely[idx] = vel[:, 1]
            self.velz[idx] = vel[:, 2]
        # can dataout be added to the pipeline too? that would remove the iterative numpy_support call.
        # could write in place on vel, which has a view of the data at dataout

        """ numpts = dataout.GetNumberOfTuples()
        numvalpts = self.probe3d.GetValidPoints().GetNumberOfTuples()
        if numpts != numvalpts:
            print(f"{numpts - numvalpts} points failed in 3d probe filter")
            msk = self.probe3d.GetOutput().GetPointData().GetArray("vtkValidPointMask")
            msk_np = numpy_support.vtk_to_numpy(msk)
            badpts = self.indices[msk_np == 0]  # this will fail once mask is not none
            for i in badpts:
                print(f"particle {i}: (x,y,z)=({self.x[i]},{self.y[i]},{self.z[i]}),
                (ux,uy,uz)=({self.velx[i]},{self.vely[i]},{self.velz[i]})")
        """

    def interp_fields(self, px=None, py=None, pz=None, twod=True, threed=True):
        """Interpolate mesh fields at current particle positions."""
        if px is None:
            px = self.x
        if py is None:
            py = self.y
        # Find current location in 2D grid and interpolate 2D fields
        if twod:
            if self.mask is None:
                self.pt2d_np[:, 0] = px
                self.pt2d_np[:, 1] = py
            else:
                idx = self.indices[self.mask]
                if idx.size != self.pt2d.GetNumberOfPoints():
                    # MOVE INTO DEACTIVATE PARTICLE -- then probe2d will always be update when we get here
                    # => assume particles are only ever deactivated, never re-activated (possible with better tracking)

                    # Number of active particles has changed, reconstruct the probe pipeline objects
                    self.pt2d_np = np.zeros((idx.size, 3))
                    self.pt2d_vtk = numpy_support.numpy_to_vtk(self.pt2d_np)
                    self.pt2d.Reset()
                    self.pt2d.SetData(self.pt2d_vtk)
                    self.ptset2d.Initialize()
                    self.ptset2d.SetPoints(self.pt2d)
                    self.probe2d.SetInputData(self.ptset2d)
                self.pt2d_np[:, 0] = px[idx]
                self.pt2d_np[:, 1] = py[idx]
            self.pt2d.Modified()
            self.probe2d.Update()

            ptsout = self.probe2d.GetOutput().GetPointData()
            elev = ptsout.GetArray("Elevation")
            wse = ptsout.GetArray("WaterSurfaceElevation")
            shear = ptsout.GetArray("ShearStress (magnitude)")
            if self.mask is None:
                self.bedelev = numpy_support.vtk_to_numpy(elev)
                self.wse = numpy_support.vtk_to_numpy(wse)
                self.shearstress = numpy_support.vtk_to_numpy(shear)
            else:
                self.bedelev[idx] = numpy_support.vtk_to_numpy(elev)
                self.wse[idx] = numpy_support.vtk_to_numpy(wse)
                self.shearstress[idx] = numpy_support.vtk_to_numpy(shear)
            if not self.track3d:
                vel = self.probe2d.GetOutput().GetPointData().GetArray("Velocity")
                vel_np = numpy_support.vtk_to_numpy(vel)
                self.velx = vel_np[:, 0]
                self.vely = vel_np[:, 1]
            self.depth = self.wse - self.bedelev
            # check shear stress (without error print statements)
            self.shearstress = np.where(self.shearstress < 0.0, 0.0, self.shearstress)
            self.ustar = (self.shearstress / 1000.0) ** 0.5

        if self.track3d and threed:
            self.normdepth = (self.z - self.bedelev) / self.depth
            # Get 3D Velocity Components
            self.interp_field_3d(px, py, pz)

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

    def is_part_wet(self, px, py, idx):
        """Determine if particle's new positions are wet.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            idx (int NumPy array): indices of particles to check

        Returns:
            wet (boolean NumPy array): True indices mean wet, False means dry
        """
        # Move this section through deactivate_particle to new function
        wet = np.full(np.size(idx), dtype=bool, fill_value=True)
        self.probe2d.Update()
        cidx = self.probe2d.GetOutput().GetPointData().GetArray("CellIndex")
        cellidx = numpy_support.vtk_to_numpy(cidx)
        """ cellidx = np.full(np.size(idx), dtype=np.int64, fill_value=-1)
        ita = np.nditer(idx, flags=["multi_index", "zerosize_ok"])
        for i in ita:
            j = ita.multi_index[0]
            point = [px[i], py[i], 0.0]
            cell.SetCellTypeToEmptyCell()
            cellidx[j] = self.mesh.CellLocator2D.FindCell(
                point, 0.0, cell, pcoords, weights
            ) """
        idxss = np.searchsorted(self.mesh.boundarycells, cellidx)
        bndcells = np.equal(self.mesh.boundarycells[idxss], cellidx)
        outofgrid = np.less(cellidx, 0)
        outparts = np.logical_or(bndcells, outofgrid)
        if outparts.any():
            b = idx[outparts]
            if self.mask is None:
                self.mask = np.full(self.nparts, fill_value=True)
            for i in np.nditer(b):
                self.deactivate_particle(i)

        # Edit the below to use probe filter

        # NOT WORKING YET
        c = np.arange(idx.size)
        c = c[~outparts[idx]]
        """ ppx = px[idx[c]]
        ppy = py[idx[c]]
        pt_np = np.zeros((c.size, 3))
        pt_np[:, 0] = ppx
        pt_np[:, 1] = ppy
        pt_vtk = numpy_support.numpy_to_vtk(pt_np)
        pt = vtk.vtkPoints()
        ptset = vtk.vtkPointSet()
        probe = vtk.vtkProbeFilter()
        strategy = vtk.vtkCellLocatorStrategy()
        pt.SetData(pt_vtk)
        ptset.SetPoints(pt)
        probe.SetInputData(ptset)
        probe.SetSourceData(self.mesh.vtksgrid2d)
        probe.SetFindCellStrategy(strategy)
        probe.Update()
        ibcvtk = probe.GetOutput().GetPointData().GetArray("IBCfp")
        ibc = numpy_support.vtk_to_numpy(ibcvtk)
        wetflgs = ibc < 0.99999
        wet[c] = wetflgs """

        cell = vtk.vtkGenericCell()
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for j in np.nditer(c, ["zerosize_ok"]):
            i = idx[j]
            point = [px[i], py[i], 0.0]
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(
                point, 0.0, cell, pcoords, weights
            )
            idlist = cell.GetPointIds()
            numpts = cell.GetNumberOfPoints()
            wet[j] = self.is_part_wet_kernel(idlist, numpts)
        return wet

    def is_part_wet_kernel(self, idlist, numpts):
        """Interpolate IBC mesh array to a point.

        Args:
            idlist (vtkIdList): list of points that define cell # cellid
            numpts (vtkIdType): number of points in idlist

        Returns:
            (bool): returns True if all points defining cell are wet, False otherwise
        """
        for i in range(numpts):
            b = self.mesh.IBC_2D.GetTuple(idlist.GetId(i))[0]
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
        px = np.copy(self.x)
        py = np.copy(self.y)

        # first perturb 2d only
        self.perturb_2d(px, py, dt)

        # check if new positions are wet or dry
        self.handle_dry_parts(px, py, dt)

        # update bed elevation, wse, depth
        self.interp_fields(px, py, threed=False)

        # Prevent particles from entering cells where depth < min_depth
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

    def perturb_random_only_2d(self, px, py, idx, dt):
        """Project new particle 2D positions based on random walk only.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            idx (int NumPy array): indices of dry particles
            dt (float): time step
        """
        px[idx] = self.x[idx] + self.xrnum[idx] * (2.0 * self.diffx[idx] * dt) ** 0.5
        py[idx] = self.y[idx] + self.yrnum[idx] * (2.0 * self.diffy[idx] * dt) ** 0.5

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
        idx = self.indices[self.depth < min_depth]
        if idx.size > 0:
            # print("particle entered min_depth")
            cell = vtk.vtkGenericCell()
            pcoords = [0.0, 0.0, 0.0]
            weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # update cell indices and interpolations
            for i in np.nditer(idx, ["zerosize_ok"]):
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
            self.depth[idx] = self.wse[idx] - self.bedelev[idx]

    def write_hdf5(self, obj, tidx, start, end, time, rank):
        """Write particle positions and interpolated quantities to file.

        Args:
            obj (h5py file): open h5py file object created with self.create_hdf5()
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
