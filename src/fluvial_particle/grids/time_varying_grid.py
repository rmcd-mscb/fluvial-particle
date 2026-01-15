"""Time-varying grid management for unsteady flow simulations."""

import pathlib
from typing import Literal

import numpy as np

from ..RiverGrid import RiverGrid


class TimeVaryingGrid:
    """Manages time-dependent flow field data with sliding window loading.

    This class wraps RiverGrid to provide support for time-varying velocity
    fields from pre-computed flow solutions. It loads grids on-demand using
    a sliding window approach (2 grids in memory at a time) and supports
    temporal interpolation between grid timesteps.

    Attributes:
        grid_times: Array of timestamps for each grid file.
        current_time: Current simulation time.
        interpolation: Interpolation method ('linear', 'nearest', 'hold').
    """

    def __init__(
        self,
        track3d: int,
        file_pattern_2d: str,
        file_pattern_3d: str,
        field_map_2d: dict,
        field_map_3d: dict,
        grid_start_index: int,
        grid_end_index: int,
        grid_dt: float,
        grid_start_time: float = 0.0,
        interpolation: Literal["linear", "nearest", "hold"] = "linear",
        min_depth: float | None = None,
        manning_n: float | None = None,
        chezy_c: float | None = None,
        darcy_f: float | None = None,
        water_density: float | None = None,
        ustar_method: str | None = None,
    ):
        """Initialize the time-varying grid manager.

        Args:
            track3d: 1 if 3D simulation, 0 for 2D only.
            file_pattern_2d: Format string for 2D grid files (e.g., "path/Result_2D_{}.vts").
            file_pattern_3d: Format string for 3D grid files (e.g., "path/Result_3D_{}.vts").
            field_map_2d: Mapping from standard field names to model-specific names for 2D.
            field_map_3d: Mapping from standard field names to model-specific names for 3D.
            grid_start_index: Index of the first grid file.
            grid_end_index: Index of the last grid file (inclusive).
            grid_dt: Time interval between grid files in seconds.
            grid_start_time: Simulation time corresponding to the first grid file.
            interpolation: Temporal interpolation method:
                - 'linear': Linear interpolation between grid timesteps.
                - 'nearest': Use the nearest grid timestep.
                - 'hold': Use the most recent grid (hold until next).
            min_depth: Minimum depth threshold for computing wet_dry.
            manning_n: Scalar Manning's n value for computing u*.
            chezy_c: Scalar Chezy C value for computing u*.
            darcy_f: Scalar Darcy-Weisbach f value for computing u*.
            water_density: Water density in kg/m³ for shear stress conversion.
            ustar_method: Force a specific u* computation method.
        """
        self.track3d = track3d
        self.file_pattern_2d = file_pattern_2d
        self.file_pattern_3d = file_pattern_3d
        self.field_map_2d = field_map_2d
        self.field_map_3d = field_map_3d
        self.grid_start_index = grid_start_index
        self.grid_end_index = grid_end_index
        self.grid_dt = grid_dt
        self.grid_start_time = grid_start_time
        self.interpolation = interpolation
        self.min_depth = min_depth
        # u* configuration options
        self._manning_n = manning_n
        self._chezy_c = chezy_c
        self._darcy_f = darcy_f
        self._water_density = water_density
        self._ustar_method = ustar_method

        # Calculate grid times
        n_grids = grid_end_index - grid_start_index + 1
        self.grid_indices = list(range(grid_start_index, grid_end_index + 1))
        self.grid_times = np.array([grid_start_time + i * grid_dt for i in range(n_grids)])

        # Validate that all grid files exist
        self._validate_grid_files()

        # Current state
        self._current_index = 0  # Index into grid_indices/grid_times
        self._current_grid: RiverGrid | None = None
        self._next_grid: RiverGrid | None = None
        self._current_time = grid_start_time

        # Load the first grid(s)
        self._load_initial_grids()

        print(
            f"TimeVaryingGrid initialized: {n_grids} timesteps, "
            f"t=[{self.grid_times[0]:.2f}, {self.grid_times[-1]:.2f}]s, "
            f"dt={grid_dt}s, interpolation={interpolation}"
        )

    def _validate_grid_files(self) -> None:
        """Validate that all grid files in the sequence exist."""
        missing_2d = []
        missing_3d = []

        for idx in self.grid_indices:
            path_2d = pathlib.Path(self.file_pattern_2d.format(idx))
            if not path_2d.exists():
                missing_2d.append(str(path_2d))

            if self.track3d:
                path_3d = pathlib.Path(self.file_pattern_3d.format(idx))
                if not path_3d.exists():
                    missing_3d.append(str(path_3d))

        if missing_2d:
            raise ValueError(f"Missing 2D grid files: {missing_2d[:3]}{'...' if len(missing_2d) > 3 else ''}")
        if missing_3d:
            raise ValueError(f"Missing 3D grid files: {missing_3d[:3]}{'...' if len(missing_3d) > 3 else ''}")

    def _load_grid_at_index(self, index: int) -> RiverGrid:
        """Load a grid at the specified index in the sequence.

        Args:
            index: Index into self.grid_indices (0-based).

        Returns:
            RiverGrid instance for the specified timestep.
        """
        file_idx = self.grid_indices[index]
        fname_2d = self.file_pattern_2d.format(file_idx)
        fname_3d = self.file_pattern_3d.format(file_idx) if self.track3d else None

        return RiverGrid(
            track3d=self.track3d,
            filename2d=fname_2d,
            filename3d=fname_3d,
            field_map_2d=self.field_map_2d,
            field_map_3d=self.field_map_3d,
            min_depth=self.min_depth,
            manning_n=self._manning_n,
            chezy_c=self._chezy_c,
            darcy_f=self._darcy_f,
            water_density=self._water_density,
            ustar_method=self._ustar_method,
        )

    def _load_initial_grids(self) -> None:
        """Load the first grid(s) at initialization."""
        self._current_grid = self._load_grid_at_index(0)

        # Load next grid if available (for interpolation)
        if len(self.grid_indices) > 1:
            self._next_grid = self._load_grid_at_index(1)

    def advance_to_time(self, t: float) -> bool:
        """Advance the grid state to the specified simulation time.

        This method checks if new grids need to be loaded based on the
        simulation time and updates the internal state accordingly.

        Args:
            t: Current simulation time in seconds.

        Returns:
            True if grids were updated, False otherwise.

        Raises:
            ValueError: If time is outside the valid grid time range.
        """
        if t < self.grid_times[0]:
            raise ValueError(f"Simulation time {t} is before first grid time {self.grid_times[0]}")

        if t > self.grid_times[-1]:
            raise ValueError(f"Simulation time {t} is after last grid time {self.grid_times[-1]}")

        self._current_time = t

        # Find which grid interval we're in
        # grid_times[new_index] <= t < grid_times[new_index + 1]
        new_index = np.searchsorted(self.grid_times, t, side="right") - 1
        new_index = max(0, min(new_index, len(self.grid_times) - 1))

        if new_index == self._current_index:
            return False  # No grid change needed

        # Need to advance grids
        grids_updated = False

        while self._current_index < new_index:
            self._current_index += 1
            grids_updated = True

            # Slide the window: next becomes current
            self._current_grid = self._next_grid

            # Load new next grid if available
            if self._current_index + 1 < len(self.grid_indices):
                self._next_grid = self._load_grid_at_index(self._current_index + 1)
                print(
                    f"Loaded grid {self.grid_indices[self._current_index + 1]} "
                    f"(t={self.grid_times[self._current_index + 1]:.2f}s)"
                )
            else:
                self._next_grid = None

        # Rebuild probe filter pipeline if grids changed
        if grids_updated and hasattr(self._current_grid, "pt2d_np"):
            nparts = self._current_grid.pt2d_np.shape[0]
            self._current_grid.build_probe_filter(nparts)
            if self._next_grid is not None:
                self._next_grid.build_probe_filter(nparts)

        return grids_updated

    def get_interpolation_weight(self) -> float:
        """Get the interpolation weight for blending current and next grids.

        Returns:
            Weight in [0, 1] where 0 = use current grid, 1 = use next grid.
            Returns 0 if using 'hold' interpolation or at last timestep.
        """
        if self.interpolation == "hold":
            return 0.0

        if self._next_grid is None:
            return 0.0

        t_current = self.grid_times[self._current_index]
        t_next = self.grid_times[self._current_index + 1]

        if self.interpolation == "nearest":
            t_mid = (t_current + t_next) / 2
            return 0.0 if self._current_time < t_mid else 1.0

        # Linear interpolation
        weight = (self._current_time - t_current) / (t_next - t_current)
        return np.clip(weight, 0.0, 1.0)

    # =========================================================================
    # Delegate methods to current grid (RiverGrid interface)
    # =========================================================================

    def build_probe_filter(self, nparts: int, comm=None) -> None:
        """Build probe filter pipeline for both current and next grids."""
        self._current_grid.build_probe_filter(nparts, comm)
        if self._next_grid is not None:
            self._next_grid.build_probe_filter(nparts, comm)

    def update_2d_pipeline(self, px, py, idx=None) -> None:
        """Update 2D probe filter pipeline."""
        weight = self.get_interpolation_weight()
        # For nearest with weight=1, only need next grid; otherwise need current
        if weight < 1.0:
            self._current_grid.update_2d_pipeline(px, py, idx)
        # For linear (any weight > 0) or nearest with weight=1, need next grid
        if self._next_grid is not None and weight > 0:
            self._next_grid.update_2d_pipeline(px, py, idx)

    def update_3d_pipeline(self, px, py, pz, idx=None) -> None:
        """Update 3D probe filter pipeline."""
        weight = self.get_interpolation_weight()
        # For nearest with weight=1, only need next grid; otherwise need current
        if weight < 1.0:
            self._current_grid.update_3d_pipeline(px, py, pz, idx)
        # For linear (any weight > 0) or nearest with weight=1, need next grid
        if self._next_grid is not None and weight > 0:
            self._next_grid.update_3d_pipeline(px, py, pz, idx)

    def out_of_grid(self, px, py, idx=None):
        """Check if particles are out of the 2D domain."""
        return self._current_grid.out_of_grid(px, py, idx)

    def reconstruct_filter_pipeline(self, nparts: int) -> None:
        """Reconstruct VTK probe filter pipeline objects."""
        self._current_grid.reconstruct_filter_pipeline(nparts)
        if self._next_grid is not None:
            self._next_grid.reconstruct_filter_pipeline(nparts)

    # =========================================================================
    # Properties delegated to current grid
    # =========================================================================

    @property
    def vtksgrid2d(self):
        """VTK structured grid for 2D data."""
        return self._current_grid.vtksgrid2d

    @property
    def vtksgrid3d(self):
        """VTK structured grid for 3D data."""
        return self._current_grid.vtksgrid3d

    @property
    def probe2d(self):
        """VTK probe filter for 2D interpolation."""
        return self._current_grid.probe2d

    @property
    def probe3d(self):
        """VTK probe filter for 3D interpolation."""
        return self._current_grid.probe3d

    @property
    def ns(self) -> int:
        """Number of stream-wise points."""
        return self._current_grid.ns

    @property
    def nn(self) -> int:
        """Number of stream-normal points."""
        return self._current_grid.nn

    @property
    def nz(self) -> int:
        """Number of vertical points."""
        return self._current_grid.nz

    @property
    def nsc(self) -> int:
        """Number of stream-wise cells."""
        return self._current_grid.nsc

    @property
    def nnc(self) -> int:
        """Number of stream-normal cells."""
        return self._current_grid.nnc

    @property
    def nzc(self) -> int:
        """Number of vertical cells."""
        return self._current_grid.nzc

    @property
    def boundarycells(self):
        """Boundary cell indices."""
        return self._current_grid.boundarycells

    # =========================================================================
    # Postprocessing (delegated to current grid)
    # =========================================================================

    def create_hdf5(self, nprints, time, fname="cells.h5", **dset_kwargs):
        """Create HDF5 file for cell-centered results."""
        return self._current_grid.create_hdf5(nprints, time, fname, **dset_kwargs)

    def postprocess(self, output_directory, n_prints, globalnparts, **dset_kwargs):
        """Write XDMF files and cumulative cell counters."""
        return self._current_grid.postprocess(output_directory, n_prints, globalnparts, **dset_kwargs)

    # =========================================================================
    # Time-varying specific properties
    # =========================================================================

    @property
    def current_time(self) -> float:
        """Current simulation time."""
        return self._current_time

    @property
    def current_grid_time(self) -> float:
        """Time of the currently loaded grid."""
        return self.grid_times[self._current_index]

    @property
    def next_grid_time(self) -> float | None:
        """Time of the next grid, or None if at last timestep."""
        if self._current_index + 1 < len(self.grid_times):
            return self.grid_times[self._current_index + 1]
        return None

    @property
    def has_next_grid(self) -> bool:
        """Whether a next grid is available for interpolation."""
        return self._next_grid is not None

    @property
    def next_probe2d(self):
        """VTK probe filter for 2D interpolation on next grid."""
        return self._next_grid.probe2d if self._next_grid else None

    @property
    def next_probe3d(self):
        """VTK probe filter for 3D interpolation on next grid."""
        return self._next_grid.probe3d if self._next_grid else None
