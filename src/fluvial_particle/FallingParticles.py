"""Falling Particles Class module."""
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
        rho=2650,
        c1=20,
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
        self.radius = radius
        self.rho = rho
        self.c1 = c1
        self.c2 = c2

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
        uz[a] = -ws  # bound below by settling velocity

        pz = self.bedelev + (self.normdepth * self.depth) + uz * dt + zranwalk
        return pz
