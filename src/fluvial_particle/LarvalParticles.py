"""LarvalParticles Class module."""
import numpy as np

from .Particles import Particles


class LarvalParticles(Particles):
    """A larval fish subclass of Particles, a helper superclass for bottom- or top-swimmers."""

    def __init__(self, nparts, x, y, z, rng, mesh, **kwargs):
        """Initialize instance of class LarvalParticles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            **kwargs (dict): additional keyword arguments  # noqa

        Optional keyword arguments:
            amp (float): amplitude of sinusoid as depth fraction, scalar or NumPy array of length nparts, optional
            period (float): period of swimming to compute ttime, scalar or NumPy array of length nparts, optional
            ttime (float): phase of swimmers, numpy array of length nparts, optional
        """
        super().__init__(nparts, x, y, z, rng, mesh, **kwargs)
        self.amp = kwargs.get("amp", 0.2)
        self.period = kwargs.get("period", 60.0)
        # Build ndarray ttime, uniformly distributed phase in [0, period]
        self.ttime = self.rng.uniform(0.0, self.period, self.nparts)

    def create_hdf5(self, nprints, globalnparts, comm=None, fname="particles.h5"):
        """Create an HDF5 file to write incremental particles results.

        Subclass override method creates additional datasets in the HDF5 file.

        Args:
            nprints (int): size of first dimension, indexes printing time slices
            globalnparts (int): global number of particles, distributed across processors
            comm (MPI communicator): only for parallel runs
            fname (string): name of the HDF5 file

        Returns:
            parts_h5: new open HDF5 file object
        """
        parts_h5 = super().create_hdf5(nprints, globalnparts, comm=comm, fname=fname)
        grp = parts_h5["properties"]
        # Subclass datasets are time invariant, only need to be written once
        grp.create_dataset("amp", (1, globalnparts), dtype=np.float64, fillvalue=np.nan)
        grp["amp"].attrs["Description"] = "amplitude of sinusoid, as fraction of depth"
        grp["amp"].attrs["Units"] = "None"
        grp.create_dataset(
            "period", (1, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grp["period"].attrs["Description"] = "period of sinusoid"
        grp["period"].attrs["Units"] = "seconds"
        grp.create_dataset(
            "ttime", (1, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grp["ttime"].attrs["Description"] = "temporal phase shift of swimmers"
        grp["ttime"].attrs["Units"] = "seconds"
        return parts_h5

    def deactivate_particles(self, idx):
        """Turn off particles that have left the river domain.

        Args:
            idx (int): index of particle to turn off
        """
        super().deactivate_particles(idx)
        if isinstance(self.amp, np.ndarray):
            self._amp[idx] = np.nan
        if isinstance(self.period, np.ndarray):
            self._period[idx] = np.nan
        if isinstance(self.ttime, np.ndarray):
            self._ttime[idx] = np.nan

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
            grp["amp"][tidx, start:end] = self.amp
            grp["period"][tidx, start:end] = self.period
            grp["ttime"][tidx, start:end] = self.ttime

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

        # Superclass attributes
        # fmt: off
        self.write_hdf5_xmf_scalarattribute(filexmf, nprints, nparts, tidx, "BedElevation", fname, "/properties/bedelev")
        self.write_hdf5_xmf_scalarattribute(filexmf, nprints, nparts, tidx, "CellIndex2D", fname, "/properties/cellidx2d")
        self.write_hdf5_xmf_scalarattribute(filexmf, nprints, nparts, tidx, "CellIndex3D", fname, "/properties/cellidx3d")
        self.write_hdf5_xmf_scalarattribute(filexmf, nprints, nparts, tidx, "Depth", fname, "/properties/depth")
        self.write_hdf5_xmf_scalarattribute(filexmf, nprints, nparts, tidx, "HeightAboveBed", fname, "/properties/htabvbed")
        self.write_hdf5_xmf_scalarattribute(filexmf, nprints, nparts, tidx, "WaterSurfaceElevation", fname, "/properties/wse")
        self.write_hdf5_xmf_vectorattribute(filexmf, nprints, nparts, tidx, "VelocityVector", fname, "/properties/velvec")

        # Subclass attributes, time invariant
        self.write_hdf5_xmf_scalarattribute(filexmf, 1, nparts, 0, "Amplitude", fname, "/properties/amp")
        self.write_hdf5_xmf_scalarattribute(filexmf, 1, nparts, 0, "Period", fname, "/properties/period")
        self.write_hdf5_xmf_scalarattribute(filexmf, 1, nparts, 0, "TimePhase", fname, "/properties/ttime")
        # fmt: on

        self.write_hdf5_xmf_gridfooter(filexmf)

    # Properties

    @property
    def amp(self):
        """[summary].

        Returns:
            [type]: [description]
        """
        return self._amp

    @amp.setter
    def amp(self, values):
        """[summary].

        Args:
            values ([type]): [description]
        """
        if isinstance(values, (int, np.int32, np.int64, float, np.float32, np.float64)):
            values = np.float64(values)
        elif isinstance(values, np.ndarray) and values.size == self.nparts:
            if values.dtype == np.float64:
                pass
            else:
                values = values.astype(np.float64)
        else:
            raise Exception(
                "amp must be either scalar or NumPy array with length = number of particles"
            )
        self._amp = values

    @property
    def period(self):
        """[summary].

        Returns:
            [type]: [description]
        """
        return self._period

    @period.setter
    def period(self, values):
        """[summary].

        Args:
            values ([type]): [description]
        """
        if isinstance(values, (int, np.int32, np.int64, float, np.float32, np.float64)):
            values = np.float64(values)
        elif isinstance(values, np.ndarray) and values.size == self.nparts:
            if values.dtype == np.float64:
                pass
            else:
                values = values.astype(np.float64)
        else:
            raise Exception(
                "period must be either scalar or NumPy array with length = number of particles"
            )
        self._period = values

    @property
    def ttime(self):
        """[summary].

        Returns:
            [type]: [description]
        """
        return self._ttime

    @ttime.setter
    def ttime(self, values):
        """[summary].

        Args:
            values ([type]): [description]
        """
        if isinstance(values, np.ndarray) and values.size == self.nparts:
            if values.dtype == np.float64:
                pass
            else:
                values = values.astype(np.float64)
        else:
            raise TypeError(
                "ttime must be NumPy array with length = number of particles"
            )
        self._ttime = values


class LarvalBotParticles(LarvalParticles):
    """A subclass of LarvalParticles for larvae that swim near bottom of water column."""

    def perturb_z(self, dt):
        """Project particles vertical trajectory, sinusoidal bed-swimmer.

        Args:
            dt (float): time step

        Returns:
            pz (float NumPy array): new elevation array
        """
        amplitude = self.depth * self.amp
        time = self.time + dt
        pz = (amplitude / 2.0) * np.sin(2.0 * np.pi * (time + self.ttime) / self.period)
        pz += self.bedelev + (amplitude / 2.0)
        self.validate_z(pz)
        return pz


class LarvalTopParticles(LarvalParticles):
    """A subclass of LarvalParticles for larvae that swim near top of water column."""

    def perturb_z(self, dt):
        """Project particles vertical trajectory, sinusoidal top-swimmer.

        Args:
            dt (float): time step

        Returns:
            pz (float NumPy array): new elevation array
        """
        amplitude = self.depth * self.amp
        time = self.time + dt
        pz = (amplitude / 2.0) * np.sin(2.0 * np.pi * (time + self.ttime) / self.period)
        pz += self.wse - (amplitude / 2.0)
        self.validate_z(pz)
        return pz
