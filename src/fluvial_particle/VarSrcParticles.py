"""Variable Source Particles Class module."""
# import numpy as np
import numpy as np

from .Helpers import load_variable_source
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
        self.sl_file = kwargs.get("StartLoc")
        self.part_start_time, x, y, z = load_variable_source(self.sl_file)
        self.start_time_mask = np.full(self.nparts, fill_value=False)
        print(len(self.part_start_time))

    @property
    def active(self):
        """Mask both in_bounds_mask and start_time_mask."""
        if self.in_bounds_mask is None:
            return self.start_time_mask
        else:
            return self.in_bounds_mask & self.start_time_mask

    def move(self, time, dt):
        """Update particle positions.

        Args:
            time (float): the new time at end of position update
            dt (float): time step
        """
        px = np.copy(self.x)
        py = np.copy(self.y)

        # Generate new random numbers
        self.gen_rands()

        # Calculate turbulent diffusion coefficients
        self.calc_diffusion_coefs()

        # Check time and compute mask
        self.start_time_mask = self.part_start_time <= self.time

        # Perturb 2D positions (and validate w.r.t. grid)
        self.perturb_2d(px, py, dt)

        # Check if new positions are wet or dry (and fix, if needed)
        self.handle_dry_parts(px, py, dt)

        # Update 2D fields
        self.interp_fields(px, py, threed=False)

        # Prevent particles from entering 2D positions where new_depth < min_depth
        self.prevent_mindepth(px, py)

        # Perturb vertical and validate
        pz = self.perturb_z(dt)

        # Move particles
        self.x = px
        self.y = py
        self.z = pz

        # Interpolate all field data at new particle positions
        self.interp_fields()

        # Update height above bed and time
        self.htabvbed = self.z - self.bedelev
        self.time.fill(time)

    def perturb_2d(self, px, py, dt):
        """Project particles' 2D trajectories based on starting interpolated quantities.

        Args:
            px (float NumPy array): new x coordinates of particles
            py (float NumPy array): new y coordinates of particles
            dt (float): time step
        """
        vx = self.velx
        vy = self.vely
        velmag = (vx**2 + vy**2) ** 0.5
        xranwalk = self.xrnum * (2.0 * self.diffx * dt) ** 0.5
        yranwalk = self.yrnum * (2.0 * self.diffy * dt) ** 0.5
        # Move and update positions in-place on each array
        a = self.indices[(velmag > 0.0) & (self.active)]
        b = self.indices[(velmag == 0.0) & (self.active)]
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
        px[b] += xranwalk[b]
        py[b] += yranwalk[b]
        self.validate_2d_pos(px, py)

    @property
    def sl_file(self) -> str:
        """Get sl_file.

        Returns:
            str: Variable source file containing start_time, x, y, z, num_particles.
        """
        return self._sl_file

    @sl_file.setter
    def sl_file(self, values):
        self._sl_file = values
