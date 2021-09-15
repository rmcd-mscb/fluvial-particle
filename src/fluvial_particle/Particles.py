"""Particles Class module."""
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
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.time = np.zeros(nparts, dtype=float)
        self.bedElev = np.zeros(nparts, dtype=float)
        self.htabvbed = np.zeros(nparts, dtype=float)
        self.wse = np.zeros(nparts, dtype=float)
        self.cellindex = np.zeros(nparts, dtype=int)  # numpy array

    def setz(self, tz):
        """Set z-value.

        Args:
            tz (float): new z-value of particle
        """
        self.z = tz

    def move_all(self, vx, vy, vz, x_diff, y_diff, z_diff, xrnum, yrnum, zrnum, dt):
        """Update position based on speed, angle.

        Args:
            vx (float): flow velocity along the x axis
            vy (float): flow velocity along the y axis
            vz (float): flow velocity along the z axis
            x_diff (float): diffusion coefficient along x
            y_diff (float): diffusion coefficient along y
            z_diff (float): diffusion coefficient along z
            xrnum (float): random number from N(0,1), scales x diffusion
            yrnum (float): random number from N(0,1), scales y diffusion
            zrnum (float): random number from N(0,1), scales z diffusion
            dt (float): time step
        """
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = xrnum * (2.0 * x_diff * dt) ** 0.5
        yranwalk = yrnum * (2.0 * y_diff * dt) ** 0.5
        zranwalk = zrnum * (2.0 * z_diff * dt) ** 0.5
        # Move and update positions in-place on each array
        a = velmag > 0.0
        self.x[a] += (
            vx[a] * dt
            + ((xranwalk[a] * vx[a]) / velmag[a])
            - ((yranwalk[a] * vy[a]) / velmag[a])
        )
        self.y[a] += (
            vy[a] * dt
            + ((xranwalk[a] * vy[a]) / velmag[a])
            + ((yranwalk[a] * vx[a]) / velmag[a])
        )
        self.z = self.z + vz * dt + zranwalk

    def move_random_only_2d(self, x_diff, y_diff, xrnum, yrnum, boolarray, dt):
        """Update position based on random walk in x and y directions.

        Args:
            x_diff ([type]): [description]
            y_diff ([type]): [description]
            xrnum ([type]): [description]
            yrnum ([type]): [description]
            boolarray ([type]): [description]
            dt ([type]): [description]
        """
        self.x[boolarray] += xrnum[boolarray] * (2.0 * x_diff[boolarray] * dt) ** 0.5
        self.y[boolarray] += yrnum[boolarray] * (2.0 * y_diff[boolarray] * dt) ** 0.5

    def project_2d(self, vx, vy, x_diff, y_diff, xrnum, yrnum, dt):
        """Forward-project new 2D position based on speed, angle.

        Args:
            vx ([type]): [description]
            vy ([type]): [description]
            x_diff ([type]): [description]
            y_diff ([type]): [description]
            xrnum ([type]): [description]
            yrnum ([type]): [description]
            dt ([type]): [description]

        Returns:
            [type]: [description]
        """
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = xrnum * (2.0 * x_diff * dt) ** 0.5
        yranwalk = yrnum * (2.0 * y_diff * dt) ** 0.5
        px = np.copy(self.x)
        py = np.copy(self.y)

        a = velmag > 0.0
        px[a] += (
            vx[a] * dt
            + ((xranwalk[a] * vx[a]) / velmag[a])
            - ((yranwalk[a] * vy[a]) / velmag[a])
        )
        py[a] += (
            vy[a] * dt
            + ((xranwalk[a] * vy[a]) / velmag[a])
            + ((yranwalk[a] * vx[a]) / velmag[a])
        )
        return px, py

    def check_z(self, alpha, bedelev, wse):
        """[summary].

        Args:
            alpha ([type]): [description]
            bedelev ([type]): [description]
            wse ([type]): [description]
        """
        # check on alpha? only makes sense for alpha<=0.5
        depth = wse - bedelev
        a = self.z > wse - alpha * depth
        b = self.z < bedelev + alpha * depth
        self.z[a] = wse[a] - alpha * depth[a]
        self.z[b] = bedelev[b] + alpha * depth[b]

    def update_info(self, cellind, time, bedelev, wse):
        """Update particle information."""
        self.bedElev = bedelev
        self.wse = wse
        self.htabvbed = self.z - self.bedElev
        self.time = time
        self.cellindex = cellind

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
