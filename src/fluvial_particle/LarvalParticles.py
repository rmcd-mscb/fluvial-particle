"""LarvalParticles Class module."""
import numpy as np

from fluvial_particle.Particles import Particles


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
            nparts ([type]): [description]
            x ([type]): [description]
            y ([type]): [description]
            z ([type]): [description]
            rng ([type]): [description]
            mesh ([type]): [description]
            track3d (int, optional): [description]. Defaults to 1.
            amp ([type]): [description]
            period ([type]): [description]
            min_elev ([type]): [description]
            ttime ([type]): [description]
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
        self._ttime = values
