"""Particle Class module."""
import math


class Particle:
    """Class of a particle."""

    def __init__(self, index, x, y, z, time_offset, amplitude, period, min_elev):
        """[summary].

        Args:
            index ([type]): [description]
            x ([type]): [description]
            y ([type]): [description]
            z ([type]): [description]
            time_offset ([type]): [description]
            amplitude ([type]): [description]
            period ([type]): [description]
            min_elev ([type]): [description]
        """
        self.x = x  # numpy array
        self.y = y  # numpy array
        self.z = z  # numpy array
        self.lastx = x  # remove
        self.lasty = y  # remove
        self.lastz = z  # remove
        self.time = 0.0
        self.bedElev = 0.0
        self.htabvbed = 0.0
        self.wse = 0.0
        self.index = index
        self.time_offset = time_offset
        self.sawtthAmplitude = amplitude
        self.sawtthPeriod = period
        self.sawtthmin_elev = min_elev
        self.vertConstElev = 0.55
        self.cellindex = 0

    def setz(self, tz):
        """Set z-value.

        Args:
            tz (float): z-value of particle
        """
        self.z = tz

    def move(self, index, vx, vy, vz, x_diff, y_diff, xrnum, yrnum, dt):
        """Update position based on speed, angle."""
        #         self.x = self.px[index-1] + (vx*dt) + xrnum*math.sqrt(2.0*x_diff*dt)
        #         self.y = self.py[index-1] + (vy*dt) + yrnum*math.sqrt(2.0*y_diff*dt)
        #         self.z = self.pz[index-1] + (vz*dt)
        velmag = math.sqrt((vx * vx) + (vy * vy))
        xranwalk = xrnum * math.sqrt(2.0 * x_diff * dt)
        yranwalk = yrnum * math.sqrt(2.0 * y_diff * dt)
        if velmag == 0:
            tmpx = self.x + (vx * dt)
            tmpy = self.y + (vy * dt)
        else:
            tmpx = (
                self.x
                + (vx * dt)
                + ((xranwalk * vx) / velmag)
                - ((yranwalk * vy) / velmag)
            )
            tmpy = (
                self.y
                + (vy * dt)
                + ((xranwalk * vy) / velmag)
                + ((yranwalk * vx) / velmag)
            )

            # tmpx = self.x + (vx*dt) + xrnum*math.sqrt(2.0*x_diff*dt)
            # tmpy = self.x + (vy*dt) + yrnum*math.sqrt(2.0*y_diff*dt)
        tmpz = self.z + (vz * dt)
        return tmpx, tmpy, tmpz

    def movewithdrift(
        self,
        index,
        vx,
        vy,
        vz,
        x_diff,
        y_diff,
        x_dgrad,
        y_dgrad,
        xrnum,
        yrnum,
        dt,
        type=1,
    ):
        """Update position based on speed, angle."""
        #         self.x = self.px[index-1] + (vx*dt) + xrnum*math.sqrt(2.0*x_diff*dt)
        #         self.y = self.py[index-1] + (vy*dt) + yrnum*math.sqrt(2.0*y_diff*dt)
        #         self.z = self.pz[index-1] + (vz*dt)
        velmag = math.sqrt((vx * vx) + (vy * vy))
        dgradmag = math.sqrt((x_dgrad * x_dgrad) + (y_dgrad * y_dgrad))
        dlong = xrnum * math.sqrt(2.0 * x_diff * dt)
        dtrans = yrnum * math.sqrt(2.0 * y_diff * dt)
        if velmag == 0:
            tmpx = self.x + ((vx + x_dgrad) * dt)
            tmpy = self.y + ((vy + y_dgrad) * dt)
        else:
            if type == 1:
                tmpx = (
                    self.x
                    + (vx * dt)
                    + x_dgrad * dt
                    + (dlong * vx / velmag - dtrans * vy / velmag)
                )
                tmpy = (
                    self.y
                    + (vy * dt)
                    + y_dgrad * dt
                    + (dlong * vy / velmag + dtrans * vx / velmag)
                )
            else:
                tmpx = (
                    self.x
                    + ((vx) * dt)
                    + (x_dgrad / dgradmag - y_dgrad / dgradmag) * dt
                    + (dlong * vx / velmag - dtrans * vy / velmag)
                )
                tmpy = (
                    self.y
                    + ((vy) * dt)
                    + (x_dgrad / dgradmag + dgradmag / dgradmag) * dt
                    + (dlong * vy / velmag + dtrans * vx / velmag)
                )

        #         tmpx = self.lastx + (vx*dt) + xrnum*math.sqrt(2.0*x_diff*dt)
        #         tmpy = self.lasty + (vy*dt) + yrnum*math.sqrt(2.0*y_diff*dt)
        tmpz = self.z + (vz * dt)
        return tmpx, tmpy, tmpz

    def move3d(
        self, index, vx, vy, vz, x_diff, y_diff, z_diff, xrnum, yrnum, zrnum, dt
    ):
        """Update position based on speed, angle."""
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
