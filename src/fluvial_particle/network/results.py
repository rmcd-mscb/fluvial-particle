"""Lazy reader for network particle output with post-processing (positions, bins, concentration, arrivals)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from .network import Network, NetworkBins
from .provider import FileHydraulicsProvider
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
        self._provider: FileHydraulicsProvider | None = None
        self._network: Network | None = None
        self._bins: dict[float, NetworkBins] = {}

    def __enter__(self) -> NetworkResults:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the particle file on context exit."""
        self.close()

    def close(self) -> None:
        """Close the particle file (and the hydraulics file if it was opened)."""
        self._ds.close()
        if self._provider is not None:
            self._provider.close()

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

    def __repr__(self) -> str:
        """One-line summary showing the file path, particle count, and output time count."""
        return f"NetworkResults({self.path}, particles={self.n_particles}, times={self.times.size})"

    @property
    def start_time(self) -> np.datetime64:
        """Run start."""
        return np.datetime64(self._ds.attrs["start_time"], "ns")

    @property
    def sources(self) -> pd.DataFrame:
        """The source table the run was configured with."""
        return pd.DataFrame(json.loads(self._ds.attrs["sources"]))

    @property
    def provider(self) -> FileHydraulicsProvider:
        """The hydraulics file the run used, reopened with the same subset and interpolation."""
        if self._provider is None:
            subset = json.loads(self._ds.attrs.get("reach_subset", "null"))
            self._provider = FileHydraulicsProvider(
                self._ds.attrs["hydraulics_file"], interpolation=self._ds.attrs["interpolation"], reach_subset=subset
            )
        return self._provider

    @property
    def network(self) -> Network:
        """Network topology and polylines from the hydraulics file."""
        if self._network is None:
            self._network = Network(self.provider.static, crs_wkt=self.provider.crs_wkt)
        return self._network

    # ---- time selection ----------------------------------------------------
    def time_index(self, time: int | np.datetime64 | str) -> int:
        """Output index for an integer (negative allowed) or the nearest datetime."""
        if isinstance(time, int | np.integer):
            return int(time) % self.times.size
        t = time.astype("datetime64[ns]") if isinstance(time, np.datetime64) else np.datetime64(str(time), "ns")
        return int(np.argmin(np.abs(self.times - t)))

    def time_indices(self, time: Any) -> npt.NDArray[np.int64]:
        """Output indices for None (all), a slice, a sequence, or a single time."""
        n = self.times.size
        if time is None:
            return np.arange(n, dtype=np.int64)
        if isinstance(time, slice):
            return np.arange(n, dtype=np.int64)[time]
        if isinstance(time, list | tuple | np.ndarray):
            return np.array([self.time_index(t) for t in time], dtype=np.int64)
        return np.array([self.time_index(time)], dtype=np.int64)

    # ---- particles -----------------------------------------------------------
    def positions(self, time: int | np.datetime64 | str | None = None) -> pd.DataFrame | xr.Dataset:
        """Particle state at one output time as a DataFrame, or the whole file as a Dataset when time is None."""
        if time is None:
            return self._ds[["reach_index", "s", "status", "mass"]]
        i = self.time_index(time)
        ri = self._ds["reach_index"].values[i].astype(np.int64)
        rid = np.where(ri >= 0, self.reach_id[np.clip(ri, 0, self.n_reach - 1)], -1)
        return pd.DataFrame({
            "particle": np.arange(self.n_particles),
            "reach_index": ri,
            "reach_id": rid,
            "s": self._ds["s"].values[i].astype(np.float64),
            "status": self._ds["status"].values[i].astype(np.int8),
            "mass": self._ds["mass"].values.astype(np.float64),
        })

    def map_positions(self, time: int | np.datetime64 | str) -> pd.DataFrame:
        """positions(time) with x, y from the polylines (NaN without polylines or when inactive)."""
        df = self.positions(time)
        assert isinstance(df, pd.DataFrame)
        x, y = self.network.map_position(df["reach_index"].to_numpy(), df["s"].to_numpy())
        df["x"] = x
        df["y"] = y
        return df

    def polylines(self) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Per-reach (x, y) vertex arrays for plotting."""
        return self.network.polylines()

    # ---- arrivals --------------------------------------------------------------
    def arrival_times(self, outlet: int | None = None) -> pd.DataFrame:
        """Exited particles with exit time, outlet, release reach, and mass; filtered to one outlet id if given."""
        ds = self._ds
        et = ds["exit_time"].values.astype(np.float64)
        er = ds["exit_reach"].values.astype(np.int64)
        done = np.isfinite(et)
        rr = ds["release_reach"].values.astype(np.int64)
        df = pd.DataFrame({
            "particle": np.nonzero(done)[0],
            "exit_time": et[done],
            "exit_datetime": self.start_time + (et[done] * 1e9).astype("timedelta64[ns]"),
            "exit_reach": er[done],
            "exit_reach_id": self.reach_id[er[done]],
            "release_reach": rr[done],
            "release_reach_id": self.reach_id[rr[done]],
            "release_time": ds["release_time"].values.astype(np.float64)[done],
            "source_index": ds["source_index"].values.astype(np.int64)[done],
            "mass": ds["mass"].values.astype(np.float64)[done],
        })
        if outlet is not None:
            df = df[df["exit_reach_id"] == int(outlet)].reset_index(drop=True)
        return df

    def arrival_histogram(self, outlet: int, bin_seconds: float) -> pd.DataFrame:
        """Mass arriving at ``outlet`` per time bin (the breakthrough curve)."""
        df = self.arrival_times(outlet)
        end = float(self.time_seconds[-1])
        edges = np.arange(0.0, end + bin_seconds, bin_seconds)
        mass, _ = np.histogram(df["exit_time"].to_numpy(), bins=edges, weights=df["mass"].to_numpy())
        starts = edges[:-1]
        return pd.DataFrame({
            "time_start": starts,
            "time": self.start_time + (starts * 1e9).astype("timedelta64[ns]"),
            "mass": mass,
        })

    def to_dataframe(self, time: int | np.datetime64 | str | None = None) -> pd.DataFrame:
        """Long-format particle state for one output time or all of them."""
        idx = self.time_indices(time)
        frames = []
        for i in idx:
            df = self.positions(int(i))
            assert isinstance(df, pd.DataFrame)
            df.insert(0, "time", self.times[i])
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def summary(self) -> str:
        """One-paragraph description of the run."""
        status = self._ds["status"].values[-1]
        exited = int((status == 2).sum())
        return (
            f"NetworkResults: {self.n_particles} particles, {self.n_reach} reaches, {self.times.size} output times "
            f"({self.times[0]} .. {self.times[-1]}), {exited} exited by the end; "
            f"mass units {self._ds.attrs.get('mass_units', '?')}; hydraulics {self._ds.attrs.get('hydraulics_file')}"
        )
