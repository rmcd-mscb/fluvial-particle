"""UserDefinedParticles Class module."""
from fluvial_particle.Particles import Particles


class UserDefinedParticles(Particles):
    """A subclass of Particles for user-defined active drift behavior."""

    def __init__(self, nparts, x, y, z, rng, mesh, track2d=0, track3d=1):
        """Standard initialization of super class.

        Args:
            nparts ([type]): [description]
            x ([type]): [description]
            y ([type]): [description]
            z ([type]): [description]
            rng ([type]): [description]
            mesh ([type]): [description]
            track2d (int, optional): [description]. Defaults to 0.
            track3d (int, optional): [description]. Defaults to 1.
        """
        super().__init__(nparts, x, y, z, rng, mesh, track2d, track3d)

    def perturb_z(self, dt):
        """[Summary].

        Args:
            dt ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Example: random walk in vertical only:
        zranwalk = self.zrnum * (2.0 * self.diffz * dt) ** 0.5
        pz = self.bedelev + (self.normdepth * self.depth) + self.velz * dt + zranwalk
        return pz
