"""Falling Particles Class module."""
import numpy as np

from .Particles import Particles


class FallingParticles(Particles):
    """A subclass of Particles that accelerate due to gravity up to the Ferguson & Church (2004) settling velocity."""

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
        radius=0.0005,
        rho=2650.0,
        c1=20.0,
        c2=1.1,
    ):
        """[Summary].

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            track3d (int): 1 if 3D model run, 0 else, optional
            lev (float): lateral eddy viscosity, scalar, optional
            beta (float): coefficients that scale diffusion, scalar or a tuple/list/numpy array of length 3, optional
            min_depth (float): minimum allowed depth that particles may enter, scalar, optional
            vertbound (float): bounds particle in fractional water column to [vertbound, 1-vertbound], scalar, optional
            comm (mpi4py object): MPI communicator used in parallel execution, optional
            radius (float): radius of the particles [m], scalar or NumPy array of length nparts, optional
            rho (float): density of the particles [kg/m^3], scalar or NumPy array of length nparts, optional
            c1 (float): viscous drag coefficient [-], scalar or NumPy array of length nparts, optional
            c2 (float): turbulent wake drag coefficient [-], scalar or NumPy array of length nparts, optional
        """
        super().__init__(
            nparts, x, y, z, rng, mesh, track3d, lev, beta, min_depth, vertbound, comm
        )
        self.c1 = c1
        self.c2 = c2
        self.radius = radius
        self.rho = rho

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
        chk1darrays = self.calc_hdf5_chunksizes(nprints)[0]
        grp = parts_h5["properties"]
        grp.create_dataset(
            "c1",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
        )
        grp["c1"].attrs["Description"] = "Viscous drag coefficient"
        grp["c1"].attrs["Units"] = "None"
        grp.create_dataset(
            "c2",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
        )
        grp["c2"].attrs["Description"] = "Turbulent wake drag coefficient"
        grp["c2"].attrs["Units"] = "None"
        grp.create_dataset(
            "radius",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
        )
        grp["radius"].attrs["Description"] = "Particle radii"
        grp["radius"].attrs["Units"] = "meters"
        grp.create_dataset(
            "rho",
            (nprints, globalnparts),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=chk1darrays,
        )
        grp["rho"].attrs["Description"] = "Particle density"
        grp["rho"].attrs["Units"] = "kilograms per cubic meter"
        return parts_h5

    def deactivate_particle(self, idx):
        """Turn off particles that have left the river domain.

        Args:
            idx (int): index of particle to turn off
        """
        super().deactivate_particle(idx)
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
            pz (float NumPy array): new elevation array
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
        grp = obj["properties"]
        grp["c1"][tidx, start:end] = self.c1
        grp["c2"][tidx, start:end] = self.c2
        grp["radius"][tidx, start:end] = self.radius
        grp["rho"][tidx, start:end] = self.rho

    def write_hdf5_xmf(self, filexmf, time, nprints, nparts, tidx):
        """Write the body of the particles XDMF file for visualizations in Paraview.

        Subclass override method writes additional attributes.

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
                <Attribute Name="c1" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/c1
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="c2" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/c2
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="Radius" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/radius
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="Density" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {tidx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{nprints} {nparts}" Format="HDF" Precision="8">
                            particles.h5:/properties/rho
                        </DataItem>
                    </DataItem>
                </Attribute>
            </Grid>"""
        )

    # Properties

    @property
    def c1(self):
        """Get c1.

        Returns:
            [type]: [description]
        """
        return self._c1

    @c1.setter
    def c1(self, values):
        """Set c1.

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
                "c1.setter wrong type, must be either float scalar or NumPy float array"
            )
        self._c1 = values

    @property
    def c2(self):
        """Get c2.

        Returns:
            [type]: [description]
        """
        return self._c2

    @c2.setter
    def c2(self, values):
        """Set c2.

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
                "c2.setter wrong type, must be either float scalar or NumPy float array"
            )
        self._c2 = values

    @property
    def radius(self):
        """Get radius.

        Returns:
            [type]: [description]
        """
        return self._radius

    @radius.setter
    def radius(self, values):
        """Set radius.

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
                "radius.setter wrong type, must be either float scalar or NumPy float array"
            )
        if np.any(values <= 0):
            raise ValueError("radius.setter must be positive number")

        self._radius = values

    @property
    def rho(self):
        """Get rho.

        Returns:
            [type]: [description]
        """
        return self._rho

    @rho.setter
    def rho(self, values):
        """Set rho.

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
                "rho.setter wrong type, must be either float scalar or NumPy float array"
            )
        if np.any(values <= 0):
            raise ValueError("rho.setter must be positive number")

        self._rho = values
