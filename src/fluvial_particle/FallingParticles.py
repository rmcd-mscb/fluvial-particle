"""FallingParticles Class module."""

import numpy as np

from .Particles import Particles


class FallingParticles(Particles):
    """A subclass of Particles that accelerate due to gravity up to the Ferguson & Church (2004) settling velocity."""

    def __init__(self, nparts, x, y, z, rng, mesh, **kwargs):
        """Initialize instance of class FallingParticles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            **kwargs (dict): additional keyword arguments  # noqa: E501

        Keyword Args:
            radius (float): radius of the particles [m], scalar or NumPy array of length nparts. Defaults to 0.0005
            rho (float): density of the particles [kg/m^3], scalar or NumPy array of length nparts. Defaults to 2650.0
            c1 (float): viscous drag coefficient [-], scalar or NumPy array of length nparts. Defaults to 20.0
            c2 (float): turbulent wake drag coefficient [-], scalar or NumPy array of length nparts. Defaults to 1.1
        """
        super().__init__(nparts, x, y, z, rng, mesh, **kwargs)
        self.c1 = kwargs.get("c1", 20.0)
        self.c2 = kwargs.get("c2", 1.1)
        self.radius = kwargs.get("radius", 0.0005)
        self.rho = kwargs.get("rho", 2650.0)

    def create_hdf5(self, nprints, globalnparts, comm=None, fname="particles.h5"):
        """Create an HDF5 file to write incremental particles results.

        Subclass override method creates additional datasets in the HDF5 file.

        Args:
            nprints (int): size of first dimension, indexes printing time slices
            globalnparts (int): global number of particles, distributed across processors
            comm (MPI communicator): only for parallel runs. Defaults to None
            fname (string): name of the HDF5 file. Defaults to "particles.h5"

        Returns:
            h5py file object: the newly created and open HDF5 file
        """
        parts_h5 = super().create_hdf5(nprints, globalnparts, comm=comm, fname=fname)
        grp = parts_h5["properties"]
        # Subclass datasets are time invariant, only need to be written once
        grp.create_dataset("c1", (1, globalnparts), dtype=np.float64, fillvalue=np.nan)
        grp["c1"].attrs["Description"] = "Viscous drag coefficient"
        grp["c1"].attrs["Units"] = "None"
        grp.create_dataset("c2", (1, globalnparts), dtype=np.float64, fillvalue=np.nan)
        grp["c2"].attrs["Description"] = "Turbulent wake drag coefficient"
        grp["c2"].attrs["Units"] = "None"
        grp.create_dataset("radius", (1, globalnparts), dtype=np.float64, fillvalue=np.nan)
        grp["radius"].attrs["Description"] = "Particle radii"
        grp["radius"].attrs["Units"] = "meters"
        grp.create_dataset("rho", (1, globalnparts), dtype=np.float64, fillvalue=np.nan)
        grp["rho"].attrs["Description"] = "Particle density"
        grp["rho"].attrs["Units"] = "kilograms per cubic meter"
        return parts_h5

    def deactivate_particles(self, idx):
        """Turn off particles that have left the river domain.

        Args:
            idx (int): index of particle to turn off
        """
        super().deactivate_particles(idx)
        if isinstance(self.c1, np.ndarray):
            self._c1[idx] = np.nan
        if isinstance(self.c2, np.ndarray):
            self._c2[idx] = np.nan
        if isinstance(self.radius, np.ndarray):
            self._radius[idx] = np.nan
        if isinstance(self.rho, np.ndarray):
            self._rho[idx] = np.nan

    def perturb_z(self, dt):
        """Project particles' vertical trajectories, random wiggle + gravitational acceleration.

        Args:
            dt (float): time step

        Returns:
            ndarray: new elevation array
        """
        z0 = self.bedelev + self.normdepth * self.depth
        zranwalk = self.zrnum * (2.0 * self.diffz * dt) ** 0.5

        g = 9.81  # magnitude of gravitational acceleration [m^2/s]
        uz = self.velz - g * dt  # accelerate the particle downwards
        specgrav = (self.rho - 1000.0) / 1000.0  # specific gravity of particle [-]
        nu = 0.000001  # kinematic viscosity of water [m^2/s]
        d = 2 * self.radius  # diameter of particle [m]
        top = specgrav * g * (d**2)
        bot = self.c1 * nu + (0.75 * self.c2 * specgrav * g * (d**3)) ** 0.5
        ws = top / bot  # Ferguson & Church (2004) settling velocity
        a = self.indices[uz < -ws]
        # bound below by settling velocity
        if isinstance(ws, np.ndarray):
            uz[a] = -ws[a]
        else:
            uz[a] = -ws
        zadv = uz * dt

        pz = z0 + zadv + zranwalk
        self.validate_z(pz)
        return pz

    def write_hdf5(self, obj, tidx, start, end, time, rank):
        """Write particle positions and interpolated quantities to file.

        Subclass override method writes additional arrays to output HDF5 file.

        Args:
            obj (h5py object): open HDF5 file object created with self.create_hdf5()
            tidx (int): current time slice index
            start (int): starting index of this processor's assigned write space
            end (int): ending write index (non-inclusive)
            time (float): current simulation time
            rank (int): processor rank if run in MPI (0 in serial)
        """
        super().write_hdf5(obj, tidx, start, end, time, rank)
        # Time invariant subclass properties
        if tidx == 0:
            grp = obj["properties"]
            grp["c1"][tidx, start:end] = self.c1
            grp["c2"][tidx, start:end] = self.c2
            grp["radius"][tidx, start:end] = self.radius
            grp["rho"][tidx, start:end] = self.rho

    def write_hdf5_xmf(self, filexmf, time, nprints, nparts, tidx):
        """Write the body of the particles XDMF file for visualizations in Paraview.

        Note that this implementation assumes the HDF5 file will be in the same directory as filexmf with the name
        particles.h5.

        Args:
            filexmf (file): open file to write
            time (float): current simulation time
            nprints (int): total number of printing steps
            nparts (int): global number of particles summed across processors
            tidx (int): time slice index corresponding to time
        """
        fname = "particles.h5"
        self.write_hdf5_xmf_gridheader(filexmf, time, nprints, nparts, tidx)

        # Superclass attributes
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
        self.write_hdf5_xmf_scalarattribute(filexmf, nprints, nparts, tidx, "Depth", fname, "/properties/depth")
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

        # Subclass attributes, time invariant
        self.write_hdf5_xmf_scalarattribute(filexmf, 1, nparts, 0, "c1", fname, "/properties/c1")
        self.write_hdf5_xmf_scalarattribute(filexmf, 1, nparts, 0, "c2", fname, "/properties/c2")
        self.write_hdf5_xmf_scalarattribute(filexmf, 1, nparts, 0, "Radius", fname, "/properties/radius")
        self.write_hdf5_xmf_scalarattribute(filexmf, 1, nparts, 0, "Density", fname, "/properties/rho")

        self.write_hdf5_xmf_gridfooter(filexmf)

    # Properties

    @property
    def c1(self):
        """np.float64 or ndarray.

        Viscous drag coef. in Ferguson and Church (2004) terminal settling velocity equation.If an ndarray,
        must be 1D and the same length as the number of simulated particles.
        """
        return self._c1

    @c1.setter
    def c1(self, values):
        if isinstance(values, (int, float, np.integer, np.floating)):
            values = np.float64(values)
        elif isinstance(values, np.ndarray) and values.size == self.nparts:
            if values.dtype == np.float64:
                pass
            else:
                values = values.astype(np.float64)
        else:
            raise Exception("c1 must be either scalar or NumPy array with length = number of particles")
        self._c1 = values

    @property
    def c2(self):
        """np.float64 or ndarray.

        Turbulent wake drag coef. in Ferguson and Church (2004) terminal settling velocity
        equation. If an ndarray, must be 1D and the same length as the number of simulated particles.
        """
        return self._c2

    @c2.setter
    def c2(self, values):
        if isinstance(values, (int, float, np.integer, np.floating)):
            values = np.float64(values)
        elif isinstance(values, np.ndarray) and values.size == self.nparts:
            if values.dtype == np.float64:
                pass
            else:
                values = values.astype(np.float64)
        else:
            raise Exception("c2 must be either scalar or NumPy array with length = number of particles")
        self._c2 = values

    @property
    def radius(self):
        """np.float64 or ndarray.

        Radius of the particles in Ferguson and Church (2004) terminal settling velocity
        equation. If an ndarray, must be 1D and the same length as the number of simulated particles.
        All radius values must be greater than 0.
        """
        return self._radius

    @radius.setter
    def radius(self, values):
        if isinstance(values, (int, float, np.integer, np.floating)):
            values = np.float64(values)
        elif isinstance(values, np.ndarray) and values.size == self.nparts:
            if values.dtype == np.float64:
                pass
            else:
                values = values.astype(np.float64)
        else:
            raise Exception("radius must be either scalar or NumPy array with length = number of particles")
        if np.any(values <= 0):
            raise ValueError("radius must be positive number")

        self._radius = values

    @property
    def rho(self):
        """np.float64 or ndarray.

        Density of the particles in Ferguson and Church (2004) terminal settling velocity
        equation. If an ndarray, must be 1D and the same length as the number of simulated particles.
        All rho values must be greater than 0.
        """
        return self._rho

    @rho.setter
    def rho(self, values):
        if isinstance(values, (int, float, np.integer, np.floating)):
            values = np.float64(values)
        elif isinstance(values, np.ndarray) and values.size == self.nparts:
            if values.dtype == np.float64:
                pass
            else:
                values = values.astype(np.float64)
        else:
            raise Exception("rho must be either scalar or NumPy array with length = number of particles")
        if np.any(values <= 0):
            raise ValueError("rho must be positive number")

        self._rho = values
