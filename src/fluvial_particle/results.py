"""Convenience classes for working with simulation results.

This module provides a high-level API for running simulations and analyzing
results, particularly useful in Jupyter notebooks.

Example usage::

    from fluvial_particle.results import run_simulation, SimulationResults

    # Run a simulation (simplest form)
    results = run_simulation("settings.py", "./output")

    # Or load existing results
    results = SimulationResults("./output")

    # Get basic info
    print(results.num_timesteps)
    print(results.num_particles)
    print(results.times)

    # Get particle positions at a specific timestep
    positions = results.get_positions(timestep=5)  # shape: (n_particles, 3)

    # Get all positions over time
    all_positions = results.get_positions()  # shape: (n_timesteps, n_particles, 3)

    # Get specific properties
    depths = results.get_property("depth", timestep=5)
    velocities = results.get_property("velvec")  # all timesteps

    # Convert to pandas DataFrame
    df = results.to_dataframe(timestep=-1)

    # Use as context manager for automatic cleanup
    with SimulationResults("./output") as results:
        positions = results.get_positions(timestep=-1)
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import h5py
import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


class SimulationResults:
    """High-level interface for accessing simulation output.

    This class provides convenient methods for loading and analyzing particle
    tracking results stored in HDF5 format.

    Attributes:
        output_dir: Path to the output directory.
        num_timesteps: Number of output timesteps in the simulation.
        num_particles: Number of particles in the simulation.
        times: Array of simulation times for each output timestep.
    """

    def __init__(self, output_dir: str | pathlib.Path) -> None:
        """Initialize results from an output directory.

        Args:
            output_dir: Path to directory containing simulation output files
                       (particles.h5, cells.h5, etc.).

        Raises:
            FileNotFoundError: If particles.h5 doesn't exist in output_dir.
        """
        self.output_dir = pathlib.Path(output_dir)
        self._particles_path = self.output_dir / "particles.h5"
        self._cells_path = self.output_dir / "cells.h5"

        if not self._particles_path.exists():
            raise FileNotFoundError(
                f"particles.h5 not found in {output_dir}. Run a simulation first or check the output directory."
            )

        self._h5file: h5py.File | None = None
        self._open()

    def _open(self) -> None:
        """Open the HDF5 file."""
        if self._h5file is None:
            self._h5file = h5py.File(self._particles_path, "r")

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def __enter__(self) -> SimulationResults:
        """Context manager entry."""
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"SimulationResults('{self.output_dir}', timesteps={self.num_timesteps}, particles={self.num_particles})"

    @property
    def num_timesteps(self) -> int:
        """Number of output timesteps."""
        return self._h5file["coordinates"]["x"].shape[0]

    @property
    def num_particles(self) -> int:
        """Number of particles."""
        return self._h5file["coordinates"]["x"].shape[1]

    @property
    def times(self) -> NDArray[np.floating]:
        """Simulation times for each output timestep."""
        times = self._h5file["coordinates"]["time"][:]
        # Handle both (n,) and (n, 1) shaped time arrays
        return times.squeeze()

    @property
    def coordinate_names(self) -> list[str]:
        """Available coordinate fields."""
        return list(self._h5file["coordinates"].keys())

    @property
    def property_names(self) -> list[str]:
        """Available property fields."""
        return list(self._h5file["properties"].keys())

    def get_positions(
        self,
        timestep: int | None = None,
        flatten_z: bool = False,
    ) -> NDArray[np.floating]:
        """Get particle positions.

        Args:
            timestep: Specific timestep index to retrieve. If None, returns
                     all timesteps. Supports negative indexing (-1 for last).
            flatten_z: If True, set z coordinates to 0 (useful for 2D visualization).

        Returns:
            If timestep is specified: array of shape (n_particles, 3)
            If timestep is None: array of shape (n_timesteps, n_particles, 3)
        """
        coords = self._h5file["coordinates"]

        if timestep is not None:
            x = coords["x"][timestep, :]
            y = coords["y"][timestep, :]
            z = coords["z"][timestep, :] if not flatten_z else np.zeros_like(x)
            return np.stack([x, y, z], axis=-1)

        x = coords["x"][:]
        y = coords["y"][:]
        z = coords["z"][:] if not flatten_z else np.zeros_like(x)
        return np.stack([x, y, z], axis=-1)

    def get_property(
        self,
        name: str,
        timestep: int | None = None,
    ) -> NDArray:
        """Get a particle property.

        Args:
            name: Property name. Use `property_names` to see available options.
                 Common properties: 'depth', 'wse', 'bedelev', 'velvec',
                 'htabvbed', 'cellidx2d', 'cellidx3d'.
            timestep: Specific timestep index. If None, returns all timesteps.

        Returns:
            Property values. Shape depends on property and timestep selection.

        Raises:
            KeyError: If property name doesn't exist.
        """
        if name not in self._h5file["properties"]:
            available = ", ".join(self.property_names)
            raise KeyError(f"Property '{name}' not found. Available: {available}")

        if timestep is not None:
            return self._h5file["properties"][name][timestep, ...]
        return self._h5file["properties"][name][:]

    def get_velocities(
        self,
        timestep: int | None = None,
    ) -> NDArray[np.floating]:
        """Get particle velocities.

        Convenience method for accessing the velocity vector property.

        Args:
            timestep: Specific timestep index. If None, returns all timesteps.

        Returns:
            Velocity vectors of shape (n_particles, 3) or (n_timesteps, n_particles, 3).
        """
        return self.get_property("velvec", timestep)

    def get_depths(
        self,
        timestep: int | None = None,
    ) -> NDArray[np.floating]:
        """Get water depths at particle locations.

        Args:
            timestep: Specific timestep index. If None, returns all timesteps.

        Returns:
            Depth values of shape (n_particles,) or (n_timesteps, n_particles).
        """
        return self.get_property("depth", timestep)

    def to_dataframe(self, timestep: int | None = None):
        """Convert results to a pandas DataFrame.

        Args:
            timestep: Specific timestep to convert. If None, includes all timesteps.

        Returns:
            pandas.DataFrame with columns for coordinates, time, and properties.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas") from err

        if timestep is not None:
            positions = self.get_positions(timestep)
            time_val = self.times[timestep]

            data = {
                "x": positions[:, 0],
                "y": positions[:, 1],
                "z": positions[:, 2],
                "time": time_val,
            }

            for prop in self.property_names:
                prop_data = self.get_property(prop, timestep)
                if prop_data.ndim == 1:
                    data[prop] = prop_data
                elif prop == "velvec":
                    data["vx"] = prop_data[:, 0]
                    data["vy"] = prop_data[:, 1]
                    data["vz"] = prop_data[:, 2]

            return pd.DataFrame(data)

        # All timesteps - create long-form dataframe
        frames = []
        for t in range(self.num_timesteps):
            df = self.to_dataframe(timestep=t)
            df["timestep"] = t
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def summary(self) -> str:
        """Return a summary of the simulation results.

        Returns:
            Multi-line string with simulation statistics.
        """
        positions = self.get_positions()
        final_positions = positions[-1]

        # Count active particles (not NaN)
        active_mask = ~np.isnan(final_positions[:, 0])
        num_active = np.sum(active_mask)

        times = self.times
        lines = [
            f"Simulation Results: {self.output_dir}",
            f"  Timesteps: {self.num_timesteps}",
            f"  Particles: {self.num_particles}",
            f"  Time range: {times[0]:.2f} - {times[-1]:.2f} s",
            f"  Active at end: {num_active} / {self.num_particles}",
            "",
            "  Coordinates: " + ", ".join(self.coordinate_names),
            "  Properties: " + ", ".join(self.property_names),
        ]

        if num_active > 0:
            final_active = final_positions[active_mask]
            lines.extend([
                "",
                "  Final position bounds (active particles):",
                f"    X: [{final_active[:, 0].min():.2f}, {final_active[:, 0].max():.2f}]",
                f"    Y: [{final_active[:, 1].min():.2f}, {final_active[:, 1].max():.2f}]",
                f"    Z: [{final_active[:, 2].min():.2f}, {final_active[:, 2].max():.2f}]",
            ])

        return "\n".join(lines)


def run_simulation(
    settings_file: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    seed: int | None = None,
    postprocess: bool = True,
    quiet: bool = False,
) -> SimulationResults:
    """Run a particle tracking simulation and return results.

    This is a convenience function that wraps the standard simulation workflow
    into a single call, useful for notebooks and scripts.

    Args:
        settings_file: Path to the user settings file (Python script).
        output_dir: Directory where output files will be written.
                   Will be created if it doesn't exist.
        seed: Random seed for reproducible simulations. If None, a seed
              is generated from current time and process ID.
        postprocess: If True (default), run post-processing to generate
                    XDMF files and cell counters.
        quiet: If True, suppress simulation output. Default False.

    Returns:
        SimulationResults object for accessing the output.

    Raises:
        FileNotFoundError: If settings_file doesn't exist.

    Example::

        from fluvial_particle.results import run_simulation

        # Run simulation and get results
        results = run_simulation("my_settings.py", "./output", seed=42)

        # Access results
        print(results.summary())
        positions = results.get_positions(timestep=-1)
    """
    import contextlib
    import io
    import time

    from .Settings import Settings
    from .simulation import simulate

    settings_path = pathlib.Path(settings_file)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    argdict = {
        "settings_file": str(settings_path),
        "output_directory": str(output_path),
        "seed": seed,
        "no_postprocess": postprocess,  # Note: confusing name, True = DO postprocess
    }

    options = Settings.read(str(settings_path))

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            simulate(options, argdict, timer=time.time)
    else:
        simulate(options, argdict, timer=time.time)

    return SimulationResults(output_path)
