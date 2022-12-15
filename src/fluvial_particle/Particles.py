"""Particles Class module."""
import h5py
import numpy as np
from vtk.util import numpy_support  # type:ignore


class Particles:
    """A class of particles, each with a velocity, size, and mass.

    Models passive particles, i.e. no active drift component.
    A superclass for particles with active drift.
    """

    def __init__(self, nparts, x, y, z, rng, mesh, **kwargs):
        """Initialize instance of class Particles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            **kwargs (dict): additional keyword arguments  # noqa

        Keyword args:
            Track3D (int): 1 if 3D model run, 0 else. Defaults to 1
            lev (float): lateral eddy viscosity, scalar. Defaults to 0.25
            beta (float): coefficients that scale diffusion, scalar or a tuple/list/numpy array of length 3. Defaults to 0.067
            min_depth (float): minimum allowed depth that particles may enter, scalar. Defaults to 0.02
            vertbound (float): bounds particle in fractional water column to [vertbound, 1-vertbound], scalar. Defaults to 0.01
            comm (mpi4py object): MPI communicator used in parallel execution. Defaults to None
            PartStartTime (float): variable particle start times. Defaults to the simulation start time
        """
        self.nparts = nparts
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.rng = rng
        self.mesh = mesh
        self.track3d = kwargs.get("Track3D", 1)
        self.lev = kwargs.get("lev", 0.25)
        self.beta = kwargs.get("beta", (0.067, 0.067, 0.067))
        self.min_depth = kwargs.get("min_depth", 0.02)
        self.vertbound = kwargs.get("vertbound", 0.01)
        self.comm = kwargs.get("comm", None)
        self._part_start_time = kwargs.get("PartStartTime", None)

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
        self._in_bounds_mask = None
        self._start_time_mask = None
        self.xrnum = np.zeros(nparts)
        self.yrnum = np.zeros(nparts)
        self.zrnum = np.zeros(nparts)

        # Construct pipeline objects for VTK probe filter (does the grid interpolations)
        self.mesh.build_probe_filter(self.nparts, self.comm)

    @property
    def active(self):
        """Mask both in_bounds_mask and start_time_mask."""
        if self.in_bounds_mask is None:
            return self.start_time_mask
        else:
            return self.in_bounds_mask & self.start_time_mask

    def calc_diffusion_coefs(self):
        """Calculate diffusion coefficients, McDonald & Nelson (2021)."""
        ustarh = self.depth * self.ustar
        self.diffx = self.lev + self.beta[0] * ustarh
        self.diffy = self.lev + self.beta[1] * ustarh
        self.diffz = self.beta[2] * ustarh

    def calc_hdf5_chunksizes(self, nprints):
        """Calculate chunksizes for datasets in particles HDF5 output.

        Designed to create chunks *close to* 1 MiB for 8 byte numbers.
        HDF5 org recommends chunks of size between 10 KiB - 1 MiB;
        Trials on Denali HPC show 1 MiB works better

        Args:
            nprints (int): total number of printing time steps

        Returns:
            tuple(tuple, tuple): 1st tuple is chunk size of datasets for 1D arrays (len=2), 2nd tuple is
            chunk size of datasets for 2D velocity arrays (len=3)
        """
        # Do the 1D particles arrays first, total write dimensions are (nprints, nparts)
        # 8 bytes per number means 1 MiB =  2^20 bytes = 2^17 numbers
        sz1mebibyte = np.int64(2**17)  # 8 bytes per number
        if self.nparts <= sz1mebibyte:
            # Case where the number of particles is less than or equal to 1 MiB
            chksz1 = np.int64(sz1mebibyte / self.nparts)
            chksz1 = np.min([chksz1, nprints])
            chksz2 = self.nparts
        else:
            # If there are > sz1mebibyte particles, chunk sizes are (1, nparts) or a subset
            chksz1 = np.int64(1)
            chksz2 = np.int64(self.nparts / np.int64(self.nparts / sz1mebibyte))
        chk1darrays = (chksz1, chksz2)

        # Now do the 2D velocity array, dimensions are (nprints, nparts, 3), or a subset
        if self.nparts <= np.int64(sz1mebibyte / 3):
            chksz1 = np.int64(sz1mebibyte / self.nparts / 3)
            chksz1 = np.min([chksz1, nprints])
            chksz2 = self.nparts
        else:
            chksz1 = np.int64(1)
            chksz2 = np.int64(self.nparts / np.int64(self.nparts / (sz1mebibyte / 3)))
        chksz3 = np.int64(3)
        chkvelarray = (chksz1, chksz2, chksz3)

        return chk1darrays, chkvelarray

    def create_hdf5(
        self, nprints, globalnparts, comm=None, fname="particles.h5", **dset_kwargs
    ):
        """Create an HDF5 file to write incremental particles results.

        Args:
            nprints (int): size of first dimension, indexes printing time slices
            globalnparts (int): global number of particles, distributed across processors
            comm (MPI communicator): only for parallel runs
            fname (string): name of the HDF5 file
            **dset_kwargs (dict): HDF5 dataset keyword arguments, e.g. compression filter # noqa

        Returns:
            h5py file object: the newly created and open HDF5 file
        """
        if comm is None:
            parts_h5 = h5py.File(fname, "w")  # Serial version
        else:
            parts_h5 = h5py.File(fname, "w", driver="mpio", comm=comm)  # MPI version

        parts_h5.attrs[
            "Description"
        ] = f"Output of the fluvial particle tracking model simulated with the {type(self).__name__} class."
        grpc = parts_h5.create_group("coordinates")
        grpc.attrs["Description"] = "Position x,y,z of particles at printing time steps"

        # Get chunk sizes
        chk1darrays, chkvelarray = self.calc_hdf5_chunksizes(nprints)
        # print("Particles HDF5 chunk sizes:", chk1darrays, ", ", chkvelarray)

        grpc.create_dataset(
            "x",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpc.create_dataset(
            "y",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpc.create_dataset(
            "z",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpc.create_dataset(
            "time", (nprints, 1), dtype=np.float64, fillvalue=np.nan, **dset_kwargs
        )
        grpc["x"].attrs["Units"] = "meters"
        grpc["y"].attrs["Units"] = "meters"
        grpc["z"].attrs["Units"] = "meters"
        grpc["time"].attrs["Units"] = "seconds"

        grpp = parts_h5.create_group("properties")
        grpp.attrs["Description"] = "Properties of particles at printing time steps"
        grpp.create_dataset(
            "bedelev",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpp.create_dataset(
            "cellidx2d",
            (nprints, globalnparts),
            dtype=np.int64,
            fillvalue=-1,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpp.create_dataset(
            "cellidx3d",
            (nprints, globalnparts),
            dtype=np.int64,
            fillvalue=-1,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpp.create_dataset(
            "depth",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpp.create_dataset(
            "htabvbed",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpp.create_dataset(
            "velvec",
            (nprints, globalnparts, 3),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chkvelarray,
            **dset_kwargs,
        )
        grpp.create_dataset(
            "wse",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
            **dset_kwargs,
        )
        grpp["bedelev"].attrs[
            "Description"
        ] = "Bed elevation, interpolated at particle positions (x,y)"
        grpp["cellidx2d"].attrs[
            "Description"
        ] = "Index of 2D grid cell containing each particle"
        grpp["cellidx3d"].attrs[
            "Description"
        ] = "Index of 3D grid cell containing each particle"
        grpp["depth"].attrs[
            "Description"
        ] = "Depth of water column, interpolated at particle positions (x,y)"
        grpp["htabvbed"].attrs[
            "Description"
        ] = "Height of particle above bed, interpolated at particle positions (x,y)"
        grpp["velvec"].attrs[
            "Description"
        ] = "Velocity vector (u,v,w), interpolated at particle positions (x,y,z) or (x,y)"
        grpp["wse"].attrs[
            "Description"
        ] = "Water surface elevation, interpolated at particle positions (x,y)"
        grpp["bedelev"].attrs["Units"] = "meters"
        grpp["cellidx2d"].attrs["Units"] = "None"
        grpp["cellidx3d"].attrs["Units"] = "None"
        grpp["depth"].attrs["Units"] = "meters"
        grpp["htabvbed"].attrs["Units"] = "meters"
        grpp["velvec"].attrs["Units"] = "meters per second"
        grpp["wse"].attrs["Units"] = "meters"
        return parts_h5

    def create_hdf5_xdmf(self, output_directory, n_prints, globalnparts):
        """Creates the particles XDMF file for visualizations in Paraview.

        Note that this implementation assumes the HDF5 file will be in the same directory as filexmf with the name particles.h5.

        Args:
            output_directory (string): path to output directory
            n_prints (int): total number of printing steps
            globalnparts (int): number of particles across all processors
        """
        parts_h5 = h5py.File(output_directory + "//particles.h5", "r")
        parts_xmf = open(output_directory + "//particles.xmf", "w")
        self.write_hdf5_xmf_header(parts_xmf)
        grpc = parts_h5["coordinates"]
        time = grpc["time"]
        gen = [t for t in time if not np.isnan(t)]
        for i in range(len(gen)):
            t = gen[i].item(0)  # this returns a python scalar, for use in f-strings
            self.write_hdf5_xmf(parts_xmf, t, n_prints, globalnparts, i)
        self.write_hdf5_xmf_footer(parts_xmf)
        parts_h5.close()
        parts_xmf.close()

    def deactivate_particles(self, idx):
        """Turn off particles that have left the river domain.

        Args:
            idx (int ndarray): index or indices of particle(s) to turn off
        """
        # Currently, particles are only ever deactivated, never re-activated
        # with better tracking, reactivation could be possible; but why do it?
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

        # nan this too?
        # self._part_start_time[idx] = np.nan

        if self.in_bounds_mask is None:
            self.in_bounds_mask = np.full(self.nparts, fill_value=True)
        self.in_bounds_mask[idx] = False

        idxx = self.indices[self.in_bounds_mask]
        if idxx.size > 0:
            # Reconstruct VTK probe filter pipeline objects
            self.mesh.reconstruct_filter_pipeline(idxx.size)

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
        # Check incoming wetness
        wet = self._is_part_wet
        if ~wet.all():
            # Move dry particles with only 2d random motion
            a = self.indices[~wet]
            self.perturb_2d_random_only(px, py, a, dt)
            # Check wetness again
            wet = self._is_part_wet
            if ~wet.all():
                # Still dry particles have no position update this step
                b = self.indices[~wet]
                px[b] = self.x[b]
                py[b] = self.y[b]

    def initial_validation(self, starttime=0.0, frac=None):
        """Validate initial 2D positions, set vertical postitions (optional), and interpolate mesh arrays.

        Args:
            starttime (float): initial time of the simulation in seconds. Defaults to 0.0
            frac (float): starting position of particles within water column (scalar or NumPy array). Defaults to None
        """
        self.validate_2d_pos(self.x, self.y)
        if self.in_bounds_mask is not None:
            if ~self.in_bounds_mask.any():
                raise Exception(
                    "No initial points in the 2D grid; check starting location(s)"
                )
            numvalidpts = self.mesh.probe2d.GetValidPoints().GetNumberOfTuples()
            inactive = self.nparts - numvalidpts
            print(f"Warning, {inactive} initial points (x, y) are not in the 2D grid.")

        # Get 2D field values
        self.interp_fields(threed=False)
        if frac is not None:
            self.z = self.bedelev + frac * self.depth
        self.validate_z(self.z)
        self.htabvbed = self.z - self.bedelev
        # Get 3D velocity field
        self.interp_fields(twod=False)

        # Set simulation start time
        self.time.fill(starttime)
        if self.part_start_time is None:
            self.part_start_time = np.full(
                self.nparts, fill_value=starttime, dtype=np.float64
            )

    def interp_3d_field(self, px=None, py=None, pz=None):
        """Interpolate 3D velocity field at current particle positions.

        Args:
            px (Numpy ndarray): Particle position coordinates. Defaults to self.x.
            py (Numpy ndarray): Particle position coordinates. Defaults to self.y.
            pz (Numpy ndarray): Particle position coordinates. Defaults to self.z.
        """
        idx = None
        if self.in_bounds_mask is not None:
            if ~self.in_bounds_mask.any():
                return
            idx = self.indices[self.in_bounds_mask]

        if px is None:
            px = self.x
        if py is None:
            py = self.y
        if pz is None:
            pz = self.z

        # Update 3D probe filter pipeline
        if idx is None:
            self.mesh.update_3d_pipeline(px, py, pz)
        else:
            self.mesh.update_3d_pipeline(px, py, pz, idx)

        # Get interpolated point data from the probe filter
        ptsout = self.mesh.probe3d.GetOutput().GetPointData()
        dataout = ptsout.GetArray("Velocity")
        vel = numpy_support.vtk_to_numpy(dataout)
        # Interpolation on cell-centered ordered integer array gives cell index number
        cellidxvtk = ptsout.GetArray("CellIndex")
        cellidx = numpy_support.vtk_to_numpy(cellidxvtk)
        if idx is None:
            self.velx = vel[:, 0]
            self.vely = vel[:, 1]
            self.velz = vel[:, 2]
            self.cellindex3d = cellidx
        else:
            self.velx[idx] = vel[:, 0]
            self.vely[idx] = vel[:, 1]
            self.velz[idx] = vel[:, 2]
            self.cellindex3d[idx] = cellidx

    def interp_fields(self, px=None, py=None, pz=None, twod=True, threed=True):
        """Interpolate mesh fields at current particle positions.

        Args:
            px (Numpy ndarray): Particle position coordinates. Defaults to self.x.
            py (Numpy ndarray): Particle position coordinates. Defaults to self.y.
            pz (Numpy ndarray): Particle position coordinates. Defaults to self.z.
            twod (bool, optional): Flag to interpolate 2D field arrays. Defaults to True.
            threed (bool, optional): Flag to interpolate 3D field arrays. Defaults to True.
        """
        idx = None
        if self.in_bounds_mask is not None:
            if ~self.in_bounds_mask.any():
                return
            idx = self.indices[self.in_bounds_mask]
        if px is None:
            px = self.x
        if py is None:
            py = self.y

        if twod:
            # Update 2D filter and interpolate 2D fields
            self.mesh.update_2d_pipeline(px, py, idx)
            ptsout = self.mesh.probe2d.GetOutput().GetPointData()
            elev = ptsout.GetArray("Elevation")
            wse = ptsout.GetArray("WaterSurfaceElevation")
            shear = ptsout.GetArray("ShearStress (magnitude)")
            # Interpolation on cell-centered ordered integer array gives cell index number
            cellidx = ptsout.GetArray("CellIndex")
            if idx is None:
                self.bedelev = numpy_support.vtk_to_numpy(elev)
                self.wse = numpy_support.vtk_to_numpy(wse)
                self.shearstress = numpy_support.vtk_to_numpy(shear)
                self.cellindex2d = numpy_support.vtk_to_numpy(cellidx)
            else:
                self.bedelev[idx] = numpy_support.vtk_to_numpy(elev)
                self.wse[idx] = numpy_support.vtk_to_numpy(wse)
                self.shearstress[idx] = numpy_support.vtk_to_numpy(shear)
                self.cellindex2d[idx] = numpy_support.vtk_to_numpy(cellidx)
            if not self.track3d:
                # Get 2D Velocity components
                vel = ptsout.GetArray("Velocity")
                vel_np = numpy_support.vtk_to_numpy(vel)
                if idx is None:
                    self.velx = vel_np[:, 0]
                    self.vely = vel_np[:, 1]
                else:
                    self.velx[idx] = vel_np[:, 0]
                    self.vely[idx] = vel_np[:, 1]
            self.depth = self.wse - self.bedelev
            self.shearstress = np.where(self.shearstress < 0.0, 0.0, self.shearstress)
            self.ustar = (self.shearstress / 1000.0) ** 0.5

        if self.track3d and threed:
            self.normdepth = (self.z - self.bedelev) / self.depth
            # Get 3D Velocity components
            self.interp_3d_field(px, py, pz)

    @property
    def _is_part_wet(self):
        """Determine if particles' new positions are wet.

        Returns:
            wet (boolean NumPy array): True indices mean wet, False means dry
        """
        # Pre-fill with True so that deactivated particles are ignored
        # check cell centered values instead
        ibcvtk = self.mesh.probe2d.GetOutput().GetPointData().GetArray("CellIBC")
        ibc = numpy_support.vtk_to_numpy(ibcvtk)
        if self.in_bounds_mask is None:
            wet = np.asarray(ibc, dtype=bool)
        else:
            wet = np.full((self.nparts,), dtype=bool, fill_value=True)
            idx = self.indices[self.in_bounds_mask]
            wet[idx] = ibc

        return wet

    def move(self, time, dt):
        """Update particle positions.

        Args:
            time (float): the new time at end of position update
            dt (float): time step
        """
        px = np.copy(self.x)
        py = np.copy(self.y)

        # Generate new random numbers
        self.gen_rands()

        # Calculate turbulent diffusion coefficients
        self.calc_diffusion_coefs()

        # Check time and compute mask
        self.start_time_mask = self.part_start_time <= self.time

        # Perturb 2D positions (and validate w.r.t. grid)
        self.perturb_2d(px, py, dt)

        # Check if new positions are wet or dry (and fix, if needed)
        self.handle_dry_parts(px, py, dt)

        # Update 2D fields
        self.interp_fields(px, py, threed=False)

        # Prevent particles from entering 2D positions where new_depth < min_depth
        self.prevent_mindepth(px, py)

        # Perturb vertical and validate
        pz = self.perturb_z(dt)

        # Move particles
        self.x = px
        self.y = py
        self.z = pz

        # Interpolate all field data at new particle positions
        self.interp_fields()

        # Update height above bed and time
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
        velmag = (vx**2 + vy**2) ** 0.5
        xranwalk = self.xrnum * (2.0 * self.diffx * dt) ** 0.5
        yranwalk = self.yrnum * (2.0 * self.diffy * dt) ** 0.5
        # Move and update positions in-place on each array
        a = self.indices[(velmag > 0.0) & self.active]
        b = self.indices[(velmag == 0.0) & self.active]
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
        self.validate_2d_pos(px, py)

    def perturb_2d_random_only(self, px, py, idx, dt):
        """Project new particle 2D positions based on random walk only.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            idx (int NumPy array): indices of dry particles
            dt (float): time step
        """
        px[idx] = self.x[idx] + self.xrnum[idx] * (2.0 * self.diffx[idx] * dt) ** 0.5
        py[idx] = self.y[idx] + self.yrnum[idx] * (2.0 * self.diffy[idx] * dt) ** 0.5
        self.validate_2d_pos(px, py)

    def perturb_z(self, dt):
        """Project particles' vertical trajectories and validate.

        Args:
            dt (float): time step

        Returns:
            ndarray: new elevation array
        """
        # Perturbations are assumed to start from the same fractional depth as last time
        z0 = self.bedelev + self.normdepth * self.depth
        zadv = self.velz * dt
        zranwalk = self.zrnum * (2.0 * self.diffz * dt) ** 0.5
        pz = z0 + zadv + zranwalk
        self.validate_z(pz)
        return pz

    def prevent_mindepth(self, px, py):
        """Prevent particles from entering a position with depth < min_depth.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
        """
        idx = self.indices[self.depth < self.min_depth]
        if idx.size > 0:
            px[idx] = self.x[idx]
            py[idx] = self.y[idx]
            # Update 2D fields
            self.interp_fields(px, py, threed=False)

    def validate_2d_pos(self, px, py):
        """Check that positions are inside the 2D grid and deactivate particles leaving it.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
        """
        if self.in_bounds_mask is None:
            idx = None
        else:
            if ~self.in_bounds_mask.any():
                return
            idx = self.indices[self.in_bounds_mask]

        # Find particles leaving the grid
        outparts = self.mesh.out_of_grid(px, py, idx)

        # Deactivate particles
        if outparts.any():
            if idx is None:
                self.deactivate_particles(outparts)
                px[outparts] = np.nan
                py[outparts] = np.nan
            else:
                self.deactivate_particles(idx[outparts])
                px[idx[outparts]] = np.nan
                py[idx[outparts]] = np.nan
            idx = self.indices[self.in_bounds_mask]
            if idx.size > 0:
                self.mesh.update_2d_pipeline(px, py, idx)

    def validate_z(self, pz):
        """Check that new particle vertical position is within bounds.

        Args:
            pz (float NumPy array): new elevation array
        """
        a = self.indices[pz > self.wse - self.vertbound * self.depth]
        b = self.indices[pz < self.bedelev + self.vertbound * self.depth]
        pz[a] = self.wse[a] - self.vertbound * self.depth[a]
        pz[b] = self.bedelev[b] + self.vertbound * self.depth[b]

    def write_hdf5(self, obj, tidx, start, end, time, rank):
        """Write particle positions and interpolated quantities to file.

        Args:
            obj (h5py object): open HDF5 file object created with self.create_hdf5()
            tidx (int): current time slice index
            start (int): starting index of this processor's assigned write space
            end (int): ending write index (non-inclusive)
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
        grpp["depth"][tidx, start:end] = self.depth
        grpp["htabvbed"][tidx, start:end] = self.htabvbed
        grpp["wse"][tidx, start:end] = self.wse
        grpp["velvec"][tidx, start:end, :] = np.vstack(
            (self.velx, self.vely, self.velz)
        ).T
        grpp["cellidx2d"][tidx, start:end] = self.cellindex2d
        grpp["cellidx3d"][tidx, start:end] = self.cellindex3d

    def write_hdf5_xmf(self, filexmf, time, nprints, nparts, tidx):
        """Write the body of the particles XDMF file for visualizations in Paraview.

        Note that this implementation assumes the HDF5 file will be in the same directory as filexmf with the name particles.h5.

        Args:
            filexmf (file): open file to write
            time (float): current simulation time
            nprints (int): total number of printing steps
            nparts (int): global number of particles summed across processors
            tidx (int): time slice index corresponding to time
        """
        fname = "particles.h5"
        self.write_hdf5_xmf_gridheader(filexmf, time, nprints, nparts, tidx)
        self.write_hdf5_xmf_scalarattribute(
            filexmf, nprints, nparts, tidx, "BedElevation", fname, "/properties/bedelev"
        )
        self.write_hdf5_xmf_scalarattribute(
            filexmf,
            nprints,
            nparts,
            tidx,
            "CellIndex2D",
            fname,
            "/properties/cellidx2d",
        )
        self.write_hdf5_xmf_scalarattribute(
            filexmf,
            nprints,
            nparts,
            tidx,
            "CellIndex3D",
            fname,
            "/properties/cellidx3d",
        )
        self.write_hdf5_xmf_scalarattribute(
            filexmf, nprints, nparts, tidx, "Depth", fname, "/properties/depth"
        )
        self.write_hdf5_xmf_scalarattribute(
            filexmf,
            nprints,
            nparts,
            tidx,
            "HeightAboveBed",
            fname,
            "/properties/htabvbed",
        )
        self.write_hdf5_xmf_scalarattribute(
            filexmf,
            nprints,
            nparts,
            tidx,
            "WaterSurfaceElevation",
            fname,
            "/properties/wse",
        )
        self.write_hdf5_xmf_vectorattribute(
            filexmf,
            nprints,
            nparts,
            tidx,
            "VelocityVector",
            fname,
            "/properties/velvec",
        )
        self.write_hdf5_xmf_gridfooter(filexmf)

    def write_hdf5_xmf_gridfooter(self, filexmf):
        """Write footer of the XDMF Uniform grid type.

        Args:
            filexmf (file): open file to write
        """
        filexmf.write("""</Grid>""")

    def write_hdf5_xmf_gridheader(self, filexmf, time, nprints, nparts, tidx):
        """Write header of the XDMF Uniform grid type.

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
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/coordinates/x
                        </DataItem>
                    </DataItem>
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                            {tidx} 0
                            1 1
                            1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/coordinates/y
                        </DataItem>
                    </DataItem>
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                            {tidx} 0
                            1 1
                            1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/coordinates/z
                        </DataItem>
                    </DataItem>
                </Geometry>
            """
        )

    def write_hdf5_xmf_scalarattribute(
        self, filexmf, nprints, nparts, tidx, name, fname, path
    ):
        """Write scalar attribute to XDMF file.

        Args:
            filexmf (file): open file to write
            nprints (int): total number of printing steps
            nparts (int): global number of particles summed across processors
            tidx (int): time slice index corresponding to time
            name (str): name of the attribute
            fname (str): name of the HDF5 file holding the data
            path (str): path to the attribute data set
        """
        filexmf.write(
            f"""<Attribute Name="{name}" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            {fname}:{path}
                        </DataItem>
                    </DataItem>
                </Attribute>
            """
        )

    def write_hdf5_xmf_vectorattribute(
        self, filexmf, nprints, nparts, tidx, name, fname, path
    ):
        """Write vector attribute to XDMF file.

        Args:
            filexmf (file): open file to write
            nprints (int): total number of printing steps
            nparts (int): global number of particles summed across processors
            tidx (int): time slice index corresponding to time
            name (str): name of the attribute
            fname (str): name of the HDF5 file holding the data
            path (str): path to the attribute data set
        """
        filexmf.write(
            f"""<Attribute Name="{name}" AttributeType="Vector" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts} 3" Format="XML">
                        <DataItem Dimensions="3 3" Format="XML">
                        {tidx} 0 0
                        1 1 1
                        1 {nparts} 3
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts} 3" Format="HDF" Precision="8">
                            {fname}:{path}
                        </DataItem>
                    </DataItem>
                </Attribute>
            """
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
        """ndarray: NumPy array of particles' bed elevation.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._bedelev

    @bedelev.setter
    def bedelev(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("bedelev.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"bedelev.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._bedelev = values

    @property
    def beta(self):
        """ndarray: diffusion scaling coefficient.

        beta accepts input as either a scalar float/np.floating, or a tuple/list/ndarray of size 3.
        beta is saved as a 3D np.float64 ndarray corresponding to (x,y,z) components.
        """
        return self._beta

    @beta.setter
    def beta(self, values):
        acceptable = (float, tuple, list, np.ndarray)
        if not isinstance(values, acceptable):
            raise TypeError(
                "beta.setter must be either a float/np.floating scalar or tuple/list/ndarray of size 3"
            )
        temp = np.zeros((3,), dtype=np.float64)
        temp[:] = values
        self._beta = temp

    @property
    def cellindex2d(self):
        """ndarray: the index of the 2D cell that each particle is currently in.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._cellindex2d

    @cellindex2d.setter
    def cellindex2d(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("cellindex2d.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"cellindex2d.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._cellindex2d = values

    @property
    def cellindex3d(self):
        """ndarray: the index of the 3D cell that each particle is currently in.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._cellindex3d

    @cellindex3d.setter
    def cellindex3d(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("cellindex3d.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"cellindex3d.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._cellindex3d = values

    @property
    def depth(self):
        """ndarray: the water-column depth that each particle is currently in.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._depth

    @depth.setter
    def depth(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("depth.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"depth.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._depth = values

    @property
    def diffx(self):
        """ndarray: the diffusion coefficient in the x-direction of each particle.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._diffx

    @diffx.setter
    def diffx(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("diffx.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"diffx.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._diffx = values

    @property
    def diffy(self):
        """ndarray: the diffusion coefficient in the y-direction of each particle.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._diffy

    @diffy.setter
    def diffy(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("diffy.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"diffy.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._diffy = values

    @property
    def diffz(self):
        """ndarray: the diffusion coefficient in the z-direction of each particle.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._diffz

    @diffz.setter
    def diffz(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("diffz.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"diffz.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._diffz = values

    @property
    def htabvbed(self):
        """ndarray: the elevation of each particle above the channel bed.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._htabvbed

    @htabvbed.setter
    def htabvbed(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("htabvbed.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"htabvbed.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._htabvbed = values

    @property
    def in_bounds_mask(self):
        """ndarray(bool): masks particles that are not in the 2D grid.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        Must have dtype=bool.
        """
        return self._in_bounds_mask

    @in_bounds_mask.setter
    def in_bounds_mask(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("in_bounds_mask requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"in_bounds_mask wrong size {values.shape}; expected ({self.nparts},)"
            )
        if not np.issubdtype(values.dtype, "bool"):
            raise TypeError("in_bounds_mask must be of 'bool' data type")
        self._in_bounds_mask = values

    @property
    def lev(self):
        """np.float64: the constant lateral eddy viscosity.

        Must be a scalar, and it will be converted to np.float64.
        """
        return self._lev

    @lev.setter
    def lev(self, values):
        if not isinstance(values, (int, float, np.floating)):
            raise TypeError("lev.setter must be a scalar")
        self._lev = np.float64(values)

    @property
    def mesh(self):
        """RiverGrid: the mesh containing the VTK grids and hydrodynamic data."""
        return self._mesh

    @mesh.setter
    def mesh(self, values):
        self._mesh = values

    @property
    def min_depth(self):
        """np.float64: the minimum water-column depth that particles may enter.

        Must be a scalar, and it will be converted to np.float64.
        """
        return self._min_depth

    @min_depth.setter
    def min_depth(self, values):
        if not isinstance(values, (int, float, np.floating)):
            raise TypeError("min_depth.setter must be a scalar")
        self._min_depth = np.float64(values)

    @property
    def normdepth(self):
        """ndarray: the normalized depth of each particle.

        In other words, the fraction of the way from the channel bed to the particle's elevation.
        In the range of (0, 1).
        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._normdepth

    @normdepth.setter
    def normdepth(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("normdepth.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"normdepth.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._normdepth = values

    @property
    def nparts(self):
        """int: the number of simulated particles.

        In parallel execution mode, this is the number of particles per CPU.
        Must be a positive integer.
        """
        return self._nparts

    @nparts.setter
    def nparts(self, values):
        if not isinstance(values, int):
            raise TypeError("nparts.setter must be int")
        if values < 1:
            raise ValueError("nparts.setter number of particles must be greater than 0")
        self._nparts = values

    @property
    def part_start_time(self):
        """np.float64 or ndarray: the starting simulation time of the particles.

        If a scalar, all particles will be activated when the simulation time exceeds part_start_time.
        If an ndarray, must be 1D and the same length as the number of simulated particles.
        """
        return self._part_start_time

    @part_start_time.setter
    def part_start_time(self, values):
        if isinstance(values, (int, float, np.integer, np.floating)):
            values = np.float64(values)
        elif isinstance(values, np.ndarray) and values.size == self.nparts:
            if values.dtype == np.float64:
                pass
            else:
                values = values.astype(np.float64)
        else:
            raise Exception(
                "part_start_time must be either scalar or NumPy array with length = number of particles"
            )

        self._part_start_time = values

    @property
    def rng(self):
        """np.random.RandomState: NumPy random number generator.

        Generates random numbers for stochastic diffusion.
        """
        return self._rng

    @rng.setter
    def rng(self, values):
        self._rng = values

    @property
    def shearstress(self):
        """ndarray: the shear stress interpolated from the mesh to each particle.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._shearstress

    @shearstress.setter
    def shearstress(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("shearstress.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"shearstress.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._shearstress = values

    @property
    def start_time_mask(self):
        """ndarray(bool): masks particles that are active by (part_start_time <= current simulation time).

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        Must have dtype=bool.
        """
        return self._start_time_mask

    @start_time_mask.setter
    def start_time_mask(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("start_time_mask requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"start_time_mask wrong size {values.shape}; expected ({self.nparts},)"
            )
        if not np.issubdtype(values.dtype, "bool"):
            raise TypeError("start_time_mask must be of 'bool' data type")
        self._start_time_mask = values

    @property
    def time(self):
        """ndarray: the current simulation time.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._time

    @time.setter
    def time(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("time.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"time.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._time = values

    @property
    def track3d(self):
        """int: flag that indicates the dimension of the simulation.

        If 3D simulation, track3d=1. Else track3d=0.
        """
        return self._track3d

    @track3d.setter
    def track3d(self, values):
        if not isinstance(values, int):
            raise TypeError("track3d.setter must be int")
        if values < 0 or values > 1:
            raise ValueError("track3d.setter must be 0 or 1")
        self._track3d = values

    @property
    def ustar(self):
        """ndarray: the shear velocity at each particle.

        Computed as the square root of shearstress / 1000 [kg/m^3].
        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._ustar

    @ustar.setter
    def ustar(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("ustar.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"ustar.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._ustar = values

    @property
    def velx(self):
        """ndarray: the fluid velocity x-component interpolated to each particle from the mesh.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._velx

    @velx.setter
    def velx(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("velx.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"velx.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._velx = values

    @property
    def vely(self):
        """ndarray: the fluid velocity y-component interpolated to each particle from the mesh.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._vely

    @vely.setter
    def vely(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("vely.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"vely.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._vely = values

    @property
    def velz(self):
        """ndarray: the fluid velocity z-component interpolated to each particle from the mesh.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._velz

    @velz.setter
    def velz(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("velz.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"velz.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._velz = values

    @property
    def vertbound(self):
        """np.float64: the fraction of the water column that buffers particles from the channel bed and water surface.

        Particles are bounded to [bedelev + vertbound * depth, wse - vertbound * depth].
        Must be between 0.0 and 0.5.
        """
        return self._vertbound

    @vertbound.setter
    def vertbound(self, values):
        if isinstance(values, (int, float, np.integer, np.floating)):
            values = np.float64(values)
        if not isinstance(values, np.float64):
            raise TypeError("vertbound.setter must be a scalar")
        if not self.track3d:
            # 2D runs have fixed position halfway up water column
            values = 0.5
        elif values >= 0.0 and values <= 0.5:
            pass
        elif values < 0.0:
            values = 0.0
            print("vertbound.setter bounded below by 0, values set to 0")
        elif values > 0.5:
            values = 0.5
            print("vertbound.setter bounded above by 0.5, values set to 0.5")
        else:
            raise ValueError("values of vertbound.setter unknown; possibly NaN?")
        self._vertbound = values

    @property
    def wse(self):
        """ndarray: the water surface elevation interpolated to each particle's 2D position.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._wse

    @wse.setter
    def wse(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("wse.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"wse.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._wse = values

    @property
    def x(self):
        """ndarray: the x-coordinate of each particle.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._x

    @x.setter
    def x(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("x.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"x.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._x = values

    @property
    def y(self):
        """ndarray: the y-coordinate of each particle.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._y

    @y.setter
    def y(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("y.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"y.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._y = values

    @property
    def z(self):
        """ndarray: the z-coordinate of each particle.

        Must be 1D, the same length as the number of particles (i.e. shape==(self.nparts,)).
        """
        return self._z

    @z.setter
    def z(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("z.setter requires a NumPy array")
        if values.shape != (self.nparts,):
            raise ValueError(
                f"z.setter wrong size {values.shape}; expected ({self.nparts},)"
            )
        self._z = values
