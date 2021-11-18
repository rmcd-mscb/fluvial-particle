"""Falling Particles Class module."""
import numpy as np

from fluvial_particle.Particles import Particles


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
            track3d (bool): 1 if 3D model run, 0 else
            radius (float): radius of the particles [m]
            rho (float): density of the particles [kg/m^3]
            c1 (float): dimensionless viscous drag coefficient
            c2 (float): dimensionless turbulent wake drag coefficient
        """
        super().__init__(nparts, x, y, z, rng, mesh, track3d)
        self.c1 = c1
        self.c2 = c2
        self.radius = radius
        self.rho = rho

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
            dt ([type]): [description]

        Returns:
            [type]: [description]
        """
        zranwalk = self.zrnum * (2.0 * self.diffz * dt) ** 0.5

        g = 9.81  # magnitude of gravitational acceleration [m^2/s]
        uz = self.velz - g * dt  # accelerate the particle downwards
        specgrav = (self.rho - 1000.0) / 1000.0  # specific gravity of particle [-]
        nu = 0.000001  # kinematic viscosity of water [m^2/s]
        d = 2 * self.radius  # diameter of particle [m]
        top = specgrav * g * (d ** 2)
        bot = self.c1 * nu + (0.75 * self.c2 * specgrav * g * (d ** 3)) ** 0.5
        ws = top / bot  # Ferguson & Church (2004) settling velocity
        a = self.indices[uz < -ws]
        # bound below by settling velocity
        if isinstance(ws, np.ndarray):
            uz[a] = -ws[a]
        else:
            uz[a] = -ws

        pz = self.bedelev + (self.normdepth * self.depth) + uz * dt + zranwalk
        return pz

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
        if isinstance(values, float):
            pass
        elif (
            isinstance(values, np.ndarray)
            and values.dtype.kind == "f"
            and values.size == self.nparts
        ):
            pass
        else:
            raise TypeError(
                "c1 wrong type, must be either float scalar or NumPy float array"
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
        if isinstance(values, float):
            pass
        elif (
            isinstance(values, np.ndarray)
            and values.dtype.kind == "f"
            and values.size == self.nparts
        ):
            pass
        else:
            raise TypeError(
                "c2 wrong type, must be either float scalar or NumPy float array"
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
        if isinstance(values, float):
            pass
        elif (
            isinstance(values, np.ndarray)
            and values.dtype.kind == "f"
            and values.size == self.nparts
        ):
            pass
        else:
            raise TypeError(
                "radius wrong type, must be either float scalar or NumPy float array"
            )
        if np.any(values <= 0):
            raise ValueError("radius must be positive number")

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
        if isinstance(values, float):
            pass
        elif (
            isinstance(values, np.ndarray)
            and values.dtype.kind == "f"
            and values.size == self.nparts
        ):
            pass
        else:
            raise TypeError(
                "rho wrong type, must be either float scalar or NumPy float array"
            )
        if np.any(values <= 0):
            raise ValueError("rho must be positive number")

        self._rho = values
