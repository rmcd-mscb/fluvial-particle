"""Variable Source Particles Class module."""
# import numpy as np
from .helpers import load_variable_source
from .Particles import Particles


class VarSrcParticles(Particles):
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
        self.sl = kwargs.get("StartLoc")
        self.part_start_time, x, y, z = load_variable_source(self.sl)
        print(len(self.part_start_time))
