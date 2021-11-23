"""LarvalParticles Class module."""
import numpy as np

from .Particles import Particles


class LarvalParticles(Particles):
    """A larval fish subclass of Particles with active drift."""

    def __init__(
        self,
        nparts,
        x,
        y,
        z,
        rng,
        mesh,
        track3d=1,
        amp=0.2,
        period=60.0,
        min_elev=0.01,
        ttime=None,
    ):
        """[summary].

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            track3d (bool): 1 if 3D model run, 0 if 2D model run
            amp (float): amplitude of sinusoidal swimming, as fraction of depth
            period (float): period of swimming, to compute ttime
            min_elev (float): minimum depth (nonfraction) that particles can enter
            ttime (float): phase of swimmers, numpy array of length nparts, optional
        """
        super().__init__(nparts, x, y, z, rng, mesh, track3d)
        self.amp = amp
        self.period = period
        self.min_elev = min_elev
        self.ttime = ttime
        # Build ndarray ttime if necessary
        if ttime is None:
            self.ttime = self.rng.uniform(0.0, self.period, self.nparts)

    def perturb_z(self, dt):
        """Project particles vertical trajectory, sinusoidal bed-swimmer.

        Args:
            dt ([type]): [description]

        Returns:
            [type]: [description]
        """
        amplitude = self.depth * self.amp
        time = self.time + dt
        pz = (amplitude / 2.0) * np.sin(2.0 * np.pi * (time + self.ttime) / self.amp)
        pz += self.bedelev + (amplitude / 2.0) + self.min_elev
        # for reference: pz = self.bedelev + (self.normdepth * self.depth) + self.velz * dt + zranwalk
        return pz

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
        self._amp = values

    @property
    def min_elev(self):
        """[summary].

        Returns:
            [type]: [description]
        """
        return self._min_elev

    @min_elev.setter
    def min_elev(self, values):
        """[summary].

        Args:
            values ([type]): [description]
        """
        self._min_elev = values

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
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "ttime.setter wrong size"
        )
        self._ttime = values
