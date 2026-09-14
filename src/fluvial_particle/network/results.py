"""Lazy reader for network particle output with post-processing (positions, bins, concentration, arrivals)."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from .writer import OUTPUT_FILENAME


class NetworkResults:
    """Open a run's ``network_particles.nc`` lazily.

    Args:
        path: the output directory or the particle file itself.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        """Open the particle file for reading.

        Args:
            path: the output directory or the particle file itself.
        """
        p = pathlib.Path(path)
        self.path = p / OUTPUT_FILENAME if p.is_dir() else p
        self._ds = xr.open_dataset(self.path, engine="h5netcdf")

    def __enter__(self) -> NetworkResults:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the particle file on context exit."""
        self.close()

    def close(self) -> None:
        """Close the particle file (and the hydraulics file if it was opened)."""
        self._ds.close()

    @property
    def attrs(self) -> dict[str, Any]:
        """Global attributes of the particle file."""
        return dict(self._ds.attrs)

    @property
    def times(self) -> npt.NDArray[np.datetime64]:
        """Output timestamps."""
        return self._ds["time"].values.astype("datetime64[ns]")

    @property
    def time_seconds(self) -> npt.NDArray[np.float64]:
        """Output times as seconds since start_time."""
        return self._ds["time_seconds"].values.astype(np.float64)

    @property
    def n_particles(self) -> int:
        """Particle count."""
        return int(self._ds.sizes["particle"])

    @property
    def n_reach(self) -> int:
        """Reach count of the run (after subsetting)."""
        return int(self._ds.sizes["reach"])

    @property
    def reach_id(self) -> npt.NDArray[np.int64]:
        """Reach ids in run order."""
        return self._ds["reach_id"].values.astype(np.int64)
