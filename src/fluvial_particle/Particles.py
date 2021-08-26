"""Particles Class module."""
import math

import numpy as np


class Particles:
    """A class of particles, each with a velocity, size, and mass."""

    def __init__(self, nparts, x, y, z, time_offset, amplitude, period, min_elev):
        """[summary].

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            time_offset (float): random offset for sinusoidal movement, numpy array of length nparts
            amplitude ([type]): [description]
            period ([type]): [description]
            min_elev ([type]): [description]
        """
        self.nparts = nparts
        self.x = x
        self.y = y
        self.z = z
        self.time_offset = time_offset
        self.time = np.zeros(nparts, dtype=float)
        self.bedElev = np.zeros(nparts, dtype=float)
        self.htabvbed = np.zeros(nparts, dtype=float)
        self.wse = np.zeros(nparts, dtype=float)
        # self.index = index
        self.sawtthAmplitude = amplitude  # numpy array if each particle can vary; scalar if constant for class instance
        self.sawtthPeriod = period  # same as amplitude
        self.sawtthmin_elev = min_elev  # same as amplitude
        self.vertConstElev = 0.55  # same as amplitude
        self.cellindex = np.zeros(nparts, dtype=int)  # numpy array

    def setz(self, tz):
        """Set z-value.

        TO BE REMOVED ?z

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

    def move3d(
        self, index, vx, vy, vz, x_diff, y_diff, z_diff, xrnum, yrnum, zrnum, dt
    ):
        """Update position based on speed, angle."""
        # TO BE REMOVED ?
        velmag = math.sqrt((vx * vx) + (vy * vy))
        dl = xrnum * math.sqrt(2.0 * x_diff * dt)
        dt = yrnum * math.sqrt(2.0 * y_diff * dt)
        dv = zrnum * math.sqrt(2.0 * z_diff * dt)
        if velmag == 0:
            tmpx = self.x + (vx * dt)
            tmpy = self.y + (vy * dt)
            tmpz = self.z + (vz * dt)
        else:
            tmpx = self.x + (vx * dt) + ((dl * vx) / velmag) - ((dt * vy) / velmag)
            tmpy = self.y + (vy * dt) + ((dl * vy) / velmag) + ((dt * vx) / velmag)
            tmpz = self.z + (vz * dt) + dv

        #         tmpx = self.lastx + (vx*dt) + xrnum*math.sqrt(2.0*x_diff*dt)
        #         tmpy = self.lasty + (vy*dt) + yrnum*math.sqrt(2.0*y_diff*dt)
        #         tmpz = self.z + (vz*dt)
        return tmpx, tmpy, tmpz

    def vert_const_depth(self, time, bedelev, wse):
        """Set verticle postion of particle at a constand elevation above the bed.

        Args:
            time ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]

        Returns:
            [type]: [description]
        """
        fishelev = self.vertConstElev
        fishelev = fishelev + bedelev
        if fishelev >= wse:
            fishelev = bedelev + (0.5 * (wse - bedelev))
        return fishelev

    def vert_mean_depth(self, time, bedelev, wse):
        """Set verticle postion of particle at the meand depth of the water column.

        Args:
            time ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]

        Returns:
            [type]: [description]
        """
        fishelev = bedelev + 0.5 * (wse - bedelev)
        return fishelev

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

    def vert_sinusoid_bottom(self, time, bedelev, wse, depth_mod):
        """Set particle postion following in a sinusoidal pattern (swim-up in larval drift).

        Particle moves as a sinusoid above the bed with within a fraction of the water-column defined
        by depth_mod.

        Args:
            time ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]
            depth_mod ([type]): [description]

        Returns:
            [type]: [description]
        """
        depth = wse - bedelev
        amplitude = depth_mod * depth
        fishelev = (amplitude / 2.0) * math.sin(
            2.0 * math.pi * (1.0 / self.sawtthPeriod) * (time + self.time_offset)
        )
        fishelev = fishelev + (amplitude / 2.0) + self.sawtthmin_elev
        fishelev = fishelev + bedelev
        if fishelev >= wse:
            if (wse - bedelev) < 0.5:
                fishelev = wse - 0.1
            else:
                fishelev = bedelev + (0.5 * (wse - bedelev))
        if fishelev <= bedelev:
            fishelev = bedelev + (0.5 * (wse - bedelev))
        return fishelev

    def vert_sinusoid_surface(self, time, bedelev, wse, depth_mod):
        """Set particle position following a sinusoidal pattern near the surface.

        Set the particle postion as a sinusoidal function with in the upper fraction of the water-column
        as defined by depth_mod.

        Args:
            time ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]
            depth_mod ([type]): [description]

        Returns:
            [type]: [description]
        """
        depth = wse - bedelev
        amplitude = depth_mod * depth
        fishelev = (amplitude / 2.0) * math.sin(
            2.0 * math.pi * (1.0 / self.sawtthPeriod) * (time + self.time_offset)
        )
        fishelev = fishelev + (amplitude / 2.0) + self.sawtthmin_elev
        fishelev = wse - fishelev
        if fishelev >= wse:
            if (wse - bedelev) < 0.5:
                fishelev = wse - 0.1
            else:
                fishelev = bedelev + (0.5 * (wse - bedelev))
        if fishelev <= bedelev:
            fishelev = bedelev + (0.5 * (wse - bedelev))
        return fishelev

    def vert_sinusoid(self, time, bedelev, wse):
        """Set the particle position above the bed folloing a sinusoidal pattern.

        Particle travels throughout the water column as a sinusoidal pattern.

        Args:
            time ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]

        Returns:
            [type]: [description]
        """
        fishelev = (self.sawtthAmplitude / 2.0) * math.sin(
            2.0 * math.pi * (1.0 / self.sawtthPeriod) * (time + self.time_offset)
        )
        fishelev = fishelev + (self.sawtthAmplitude / 2.0) + self.sawtthmin_elev
        fishelev = fishelev + bedelev
        if fishelev >= wse:
            if (wse - bedelev) < 0.5:
                fishelev = wse - 0.1
            else:
                fishelev = bedelev + (0.5 * (wse - bedelev))
        if fishelev <= bedelev:
            fishelev = bedelev + (0.5 * (wse - bedelev))
        return fishelev

    def vert_sawtooth(self, time, bedelev, wse):
        """Set the particle postion above the bed as a sawtooth pattern.

        Args:
            time ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]

        Returns:
            [type]: [description]
        """
        # http://en.wikipedia.org/wiki/Sawtooth_function
        offset = (0.5 * self.sawtthAmplitude) + self.sawtthmin_elev
        fishelev = (
            -1.0
            * self.sawtthAmplitude
            * (
                (time / self.sawtthPeriod)
                - math.floor(0.5 + (time / self.sawtthPeriod))
            )
            + offset
        )
        fishelev = fishelev + bedelev
        if fishelev >= wse:
            if (wse - bedelev) < 0.5:
                fishelev = wse - 0.1
            else:
                fishelev = bedelev + (0.5 * (wse - bedelev))
        if fishelev <= bedelev:
            fishelev = bedelev + (0.5 * (wse - bedelev))
        return fishelev

    def move_random_only_2d(self, index, x_diff, y_diff, xrnum, yrnum, dt):
        """Update position based on speed, angle."""
        #         self.x = self.px[index-1] + xrnum*math.sqrt(2.0*x_diff*dt)
        #         self.y = self.py[index-1] + yrnum*math.sqrt(2.0*y_diff*dt)
        #         self.z = self.pz[index-1]
        tmpx = self.x + xrnum * math.sqrt(2.0 * x_diff * dt)
        tmpy = self.y + yrnum * math.sqrt(2.0 * y_diff * dt)
        tmpz = self.z
        #         print 'in random x_diff: %f y_diff: %f' %(xrnum*math.sqrt(2.0*x_diff*dt),yrnum*math.sqrt(2.0*y_diff*dt))
        return tmpx, tmpy, tmpz

    def move_random_only_3d(
        self, index, x_diff, y_diff, z_diff, xrnum, yrnum, zrnum, dt
    ):
        """Update position based on speed, angle."""
        tmpx = self.x + xrnum * math.sqrt(2.0 * x_diff * dt)
        tmpy = self.y + yrnum * math.sqrt(2.0 * y_diff * dt)
        tmpz = self.z + zrnum * math.sqrt(2.0 * z_diff * dt)
        #         print 'in random x_diff: %f y_diff: %f' %(xrnum*math.sqrt(2.0*x_diff*dt),yrnum*math.sqrt(2.0*y_diff*dt))
        return tmpx, tmpy, tmpz

    def update_position(self, index, cellind, xt, yt, zt, time, bedelev, wse):
        """Update position of particle."""
        #        assert index == len(self.px)+1, "index is not correct: "
        #         self.px.append(x)
        #         self.py.append(y)
        #         self.pz.append(z)
        self.lastx = self.x
        self.lasty = self.y
        self.lastz = self.z
        self.x = xt
        self.y = yt
        self.z = zt
        self.bedElev = bedelev
        self.wse = wse
        self.htabvbed = self.z - self.bedElev
        self.time = time
        self.cellindex = cellind
        if self.wse <= self.bedElev:
            #           self.z = self.bedElev+0.5(self.wse-self.bedElev)
            print(f"error: index {index}: z: {zt} bed: {bedelev} wse: {wse}")

    def keep_postition(self, time):
        """Keep position of particle.

        Args:
            time ([type]): [description]
        """
        self.lastx = self.x
        self.lasty = self.y
        self.lastz = self.z
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
            self.index,
            self.time,
            self.cellindex,
            self.x,
            self.y,
            self.z,
            self.bedElev,
            self.htabvbed,
            self.wse,
        )


#     def getIPosition(self, index):
#         return self.index, self.time[index], self.px[index], self.py[index], self.pz[index], self.bedElev[index], self.wse[index]
#
#     def experienceDrag(self):
#         """ Slow particle down through drag """
#         self.speed *= self.drag
#
#     def accelerate(self, vector):
#         """ Change angle and speed by a given vector """
#         (self.angle, self.speed) = addVectors((self.angle, self.speed), vector)
