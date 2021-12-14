"""LarvalParticles Class module."""
import numpy as np

from .Particles import Particles


class LarvalParticles(Particles):
    """A larval fish subclass of Particles.

    A superclass for bottom-swimming larvae (LarvalBotParticles) & top-swimming larvae (LarvalTopParticles).
    """

    def __init__(
        self,
        nparts,
        x,
        y,
        z,
        rng,
        mesh,
        track3d=1,
        lev=0.25,
        beta=(0.067, 0.067, 0.067),
        min_depth=0.02,
        vertbound=0.01,
        comm=None,
        amp=0.2,
        period=60.0,
        ttime=None,
    ):
        """Initialize instance of class LarvalTopParticles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            track3d (int): 1 if 3D model run, 0 if 2D model run, optional
            lev (float): lateral eddy viscosity, scalar, optional
            beta (float): coefficients that scale diffusion, scalar or a tuple/list/numpy array of length 3, optional
            min_depth (float): minimum allowed depth that particles may enter, scalar, optional
            vertbound (float): bounds particle in fractional water column to [vertbound, 1-vertbound], scalar, optional
            comm (mpi4py object): MPI communicator used in parallel execution, optional
            amp (float): amplitude of sinusoid as depth fraction, scalar or NumPy array of length nparts, optional
            period (float): period of swimming to compute ttime, scalar or NumPy array of length nparts, optional
            ttime (float): phase of swimmers, numpy array of length nparts, optional
        """
        super().__init__(
            nparts, x, y, z, rng, mesh, track3d, lev, beta, min_depth, vertbound, comm
        )
        self.amp = amp
        self.period = period
        self.ttime = ttime
        # Build ndarray ttime if necessary
        if ttime is None:
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
        grp.create_dataset(
            "amp", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grp["amp"].attrs[
            "Description"
        ] = "amplitude of sinusoidal swimming, as fraction of depth"
        grp["amp"].attrs["Units"] = "None"
        grp.create_dataset(
            "period", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grp["period"].attrs["Description"] = "period of swimming, to compute ttime"
        grp["period"].attrs["Units"] = "seconds"
        grp.create_dataset(
            "ttime", (nprints, globalnparts), dtype=np.float64, fillvalue=np.nan
        )
        grp["ttime"].attrs[
            "Description"
        ] = "phase of swimmers, numpy array of length nparts"
        grp["ttime"].attrs["Units"] = "None"
        return parts_h5

    def deactivate_particle(self, idx):
        """Turn off particles that have left the river domain.

        Args:
            idx (int): index of particle to turn off
        """
        super().deactivate_particle(idx)
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
        grp = obj["properties"]
        grp["amp"][tidx, start:end] = self.amp
        grp["period"][tidx, start:end] = self.period
        grp["ttime"][tidx, start:end] = self.ttime

    def write_hdf5_xmf(self, filexmf, time, nprints, nparts, tidx):
        """Write the body of the particles XDMF file for visualizations in Paraview.

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
                <Attribute Name="BedElevation" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
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
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
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
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/cellidx3d
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="Depth" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/depth
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
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
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
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
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
                        <DataItem Dimensions="{nprints} {nparts} 3" Format="HDF" Precision="8">
                            particles.h5:/properties/velvec
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="Amplitude" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/amp
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="Period" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/period
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="TimePhase" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/ttime
                        </DataItem>
                    </DataItem>
                </Attribute>
            </Grid>"""
        )

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
        if isinstance(values, int):
            values = np.float64(values)
        elif isinstance(values, float):
            pass
        elif (
            isinstance(values, np.ndarray)
            and values.dtype.kind == "f"
            and values.size == self.nparts
        ):
            pass
        else:
            raise TypeError(
                "amp.setter wrong type, must be either float scalar or NumPy float array"
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
        self._period = values
        if isinstance(values, int):
            values = np.float64(values)
        elif isinstance(values, float):
            pass
        elif (
            isinstance(values, np.ndarray)
            and values.dtype.kind == "f"
            and values.size == self.nparts
        ):
            pass
        else:
            raise TypeError(
                "period.setter wrong type, must be either float scalar or NumPy float array"
            )

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
        if (
            isinstance(values, np.ndarray)
            and values.dtype.kind == "f"
            and values.size == self.nparts
        ):
            pass
        else:
            raise TypeError(
                "ttime.setter wrong type, must be NumPy float array of length nparts"
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
        pz = (amplitude / 2.0) * np.sin(2.0 * np.pi * (time + self.ttime) / self.amp)
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
        pz = (amplitude / 2.0) * np.sin(2.0 * np.pi * (time + self.ttime) / self.amp)
        pz += self.wse - (amplitude / 2.0)
        self.validate_z(pz)
        return pz
