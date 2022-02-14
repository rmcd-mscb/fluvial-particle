"""Variable Source Particles Class module."""
import numpy as np

from .Particles import Particles

class VSParticles(Particles):
    """Variable Source Particles.

    Args:
        Particles ([type]): [description]
    """

        def __init__(self, nparts, x, y, z, rng, mesh, **kwargs):
        """Initialize instance of class FallingParticles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            **kwargs (dict): additional keyword arguments  # noqa

        Optional keyword arguments:
            radius (float): radius of the particles [m], scalar or NumPy array of length nparts, optional
            rho (float): density of the particles [kg/m^3], scalar or NumPy array of length nparts, optional
            c1 (float): viscous drag coefficient [-], scalar or NumPy array of length nparts, optional
            c2 (float): turbulent wake drag coefficient [-], scalar or NumPy array of length nparts, optional
        """
        super().__init__(nparts, x, y, z, rng, mesh, **kwargs)
        self.c1 = kwargs.get("c1", 20.0)
        self.c2 = kwargs.get("c2", 1.1)
        self.radius = kwargs.get("radius", 0.0005)
        self.rho = kwargs.get("rho", 2650.0)
