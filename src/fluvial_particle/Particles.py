"""Particles Class module."""
import math

import numpy as np


class Particles:
    """A class of particles, each with a velocity, size, and mass."""

    def __init__(self, nparts, x, y, z):
        """Initialize instance of class Particles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
        """
        self.nparts = nparts
        self.x = x
        self.y = y
        self.z = z
        self.time = np.zeros(nparts, dtype=float)
        self.bedElev = np.zeros(nparts, dtype=float)
        self.htabvbed = np.zeros(nparts, dtype=float)
        self.wse = np.zeros(nparts, dtype=float)
        self.cellindex = np.zeros(nparts, dtype=int)  # numpy array

    def setz(self, tz):
        """Set z-value.

        TO BE REMOVED ?

        Args:
            tz (float): z-value of particle
        """
        self.z = tz

    def move(self, vx, vy, vz, x_diff, y_diff, xrnum, yrnum, dt):
        """Update position based on speed, angle.

        Args:
            vx (float): velocity along x-coordinate axis [m/s]
            vy (float): velocity along y-coordinate axis [m/s]
            vz (float): velocity along z-coordinate axis [m/s]
            x_diff (float): diffusion coefficient (>=0) along x-coordinate axis [m^2/s]
            y_diff (float): diffusion coefficient (>=0) along y-coordinate axis [m^2/s]
            xrnum (float): drawn from standard normal distribution, scales x random walk
            yrnum (float): drawn from standard normal distribution, scales y random walk
            dt (float): time step [s]
        """
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = xrnum * (2.0 * x_diff * dt) ** 0.5
        yranwalk = yrnum * (2.0 * y_diff * dt) ** 0.5

        # numpy conditional evaluations
        # Move and update positions in-place on each array
        self.x = self.x + np.where(
            velmag > 0.0,
            vx * dt + ((xranwalk * vx) / velmag) - ((yranwalk * vy) / velmag),
            0.0,
        )
        self.y = self.y + np.where(
            velmag > 0.0,
            vy * dt + ((xranwalk * vy) / velmag) + ((yranwalk * vx) / velmag),
            0.0,
        )
        self.z = self.z + (vz * dt)

        # if velmag == 0:
        #    tmpx = self.x + (vx * dt)
        #    tmpy = self.y + (vy * dt)
        # else:
        #    tmpx = (
        #        self.x
        #        + (vx * dt)
        #        + ((xranwalk * vx) / velmag)
        #        - ((yranwalk * vy) / velmag)
        #    )
        #    tmpy = (
        #        self.y
        #        + (vy * dt)
        #        + ((xranwalk * vy) / velmag)
        #        + ((yranwalk * vx) / velmag)
        #    )

    def vert_random_walk(self, time, bedelev, wse, vx, vy, vz, z_diff, zrnum, dt):
        """Set particle postion as a random walk above the bed.

        Args:
            time ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]
            vx ([type]): [description]
            vy ([type]): [description]
            vz ([type]): [description]
            z_diff ([type]): [description]
            zrnum ([type]): [description]
            dt ([type]): [description]

        Returns:
            [type]: [description]
        """
        velmag = math.sqrt((vx * vx) + (vy * vy))
        dv = zrnum * math.sqrt(2.0 * z_diff * dt)
        tdepth = wse - bedelev
        #        tmpz = bedelev
        if velmag == 0:
            tmpz = self.z + (vz * dt)
        else:
            tmpz = self.z + (vz * dt) + dv

        if tmpz >= wse:
            # tmpz = (wse - (0.05*tdepth))
            tmpz = wse - 0.01 * tdepth
        if tmpz <= bedelev:
            # tmpz = (wse - (0.05*tdepth))
            tmpz = bedelev + 0.01 * tdepth
        return tmpz

    def move_random_only_2d(self, x_diff, y_diff, xrnum, yrnum, dt):
        """Update position based on speed, angle."""
        tmpx = self.x + xrnum * math.sqrt(2.0 * x_diff * dt)
        tmpy = self.y + yrnum * math.sqrt(2.0 * y_diff * dt)
        tmpz = self.z
        return tmpx, tmpy, tmpz

    def move_random_only_3d(self, x_diff, y_diff, z_diff, xrnum, yrnum, zrnum, dt):
        """Update position based on speed, angle."""
        tmpx = self.x + xrnum * math.sqrt(2.0 * x_diff * dt)
        tmpy = self.y + yrnum * math.sqrt(2.0 * y_diff * dt)
        tmpz = self.z + zrnum * math.sqrt(2.0 * z_diff * dt)
        return tmpx, tmpy, tmpz

    def update_position(self, cellind, xt, yt, zt, time, bedelev, wse):
        """Update position of particle."""
        self.x = xt
        self.y = yt
        self.z = zt
        self.bedElev = bedelev
        self.wse = wse
        self.htabvbed = self.z - self.bedElev
        self.time = time
        self.cellindex = cellind
        if self.wse <= self.bedElev:
            print(f"error: z: {zt} bed: {bedelev} wse: {wse}")

    def keep_postition(self, time):
        """Keep position of particle.

        Args:
            time ([type]): [description]
        """
        self.time = time
        if self.z <= self.bedElev:
            self.z = self.bedElev + 0.5 * (self.wse - self.bedElev)

    #            print "error: index %i: z: %f bed: %f wse: %f" %(index, z, bedelev, wse)
    def get_position(self):
        """Return position of particle."""
        return self.x, self.y, self.z

    def get_total_position(self):
        """Return complete position of particle."""
        return (
            self.time,
            self.cellindex,
            self.x,
            self.y,
            self.z,
            self.bedElev,
            self.htabvbed,
            self.wse,
        )
