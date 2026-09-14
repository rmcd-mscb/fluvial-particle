"""Lazy reader for network particle output with post-processing (positions, bins, concentration, arrivals)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from ..io import PVDWriter, VTPWriter
from .dispersion import dispersion_coefficient
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
        """The hydraulics file the run used, reopened with the same subset, interpolation, and dtype."""
        if self._provider is None:
            subset = json.loads(self._ds.attrs.get("reach_subset", "null"))
            self._provider = FileHydraulicsProvider(
                self._ds.attrs["hydraulics_file"],
                interpolation=self._ds.attrs["interpolation"],
                dtype=self._ds.attrs.get("dtype", "float64"),
                reach_subset=subset,
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
        """Output index for an integer (negative allowed) or the nearest datetime.

        Raises:
            IndexError: an integer ``time`` is out of range for the output time axis.
        """
        if isinstance(time, int | np.integer):
            n = self.times.size
            i = int(time)
            if not -n <= i < n:
                raise IndexError(f"time index {i} is out of range for {n} output times")
            return i % n
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
        """positions(time) with x, y from the polylines.

        A reach with a single vertex maps to that vertex and one with no vertices to the static
        x_mid/y_mid, so x and y are NaN only for an inactive particle or for a reach with neither
        polyline nor midpoint.
        """
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

    # ---- bins and concentration ---------------------------------------------
    def bins(self, bin_length: float = 100.0) -> NetworkBins:
        """Sub-reach bins of about ``bin_length`` meters (cached per length; np.inf gives one bin per reach)."""
        key = float(bin_length)
        if key not in self._bins:
            self._bins[key] = NetworkBins(self.network, key)
        return self._bins[key]

    def _bandwidth(self, i: int, _bins: NetworkBins, smoothing: float | str | None) -> npt.NDArray[np.float64] | None:
        if smoothing is None:
            return None
        if isinstance(smoothing, str):
            if smoothing != "auto":
                raise ValueError("smoothing must be None, a bandwidth in meters, or 'auto'")
            d = json.loads(self._ds.attrs["dispersion"])
            h = self.provider.hydraulics(self.times[i])
            k = dispersion_coefficient(h, d["model"], scale=d["scale"], cap=d["cap"], value=d["value"])
            return np.sqrt(2.0 * k * float(self._ds.attrs["dt"]))
        return np.full(self.n_reach, float(smoothing))

    def _mass_per_bin(
        self, i: int, bins: NetworkBins, weights: npt.NDArray[np.float64], smoothing: float | str | None
    ) -> npt.NDArray[np.float64]:
        ri = self._ds["reach_index"].values[i].astype(np.int64)
        si = self._ds["s"].values[i].astype(np.float64)
        act = self._ds["status"].values[i] == 1
        r, s, w = ri[act], si[act], weights[act]
        bw = self._bandwidth(i, bins, smoothing)
        if bw is None:
            return np.bincount(bins.bin_of(r, s), weights=w, minlength=bins.n_bins).astype(np.float64)
        out = np.zeros(bins.n_bins)
        centers = 0.5 * (bins.s_start + bins.s_end)
        for reach in np.unique(r):
            sel = r == reach
            b0 = int(bins.reach_bin_start[reach])
            nb = int(bins.bins_per_reach[reach])
            h = max(float(bw[reach]), 1e-6)
            d = centers[b0 : b0 + nb][None, :] - s[sel][:, None]
            kern = np.exp(-0.5 * (d / h) ** 2)
            empty = np.nonzero(kern.sum(axis=1) == 0.0)[0]  # bandwidth far below the bin width: nearest bin
            kern[empty, np.argmin(np.abs(d[empty]), axis=1)] = 1.0
            kern /= kern.sum(axis=1, keepdims=True)
            out[b0 : b0 + nb] += (kern * w[sel][:, None]).sum(axis=0)
        return out

    def _bin_coords(self, bins: NetworkBins) -> dict[str, Any]:
        return {
            "bin": np.arange(bins.n_bins),
            "bin_reach": ("bin", bins.bin_reach),
            "reach_id": ("bin", self.reach_id[bins.bin_reach]),
            "s_start": ("bin", bins.s_start),
            "s_end": ("bin", bins.s_end),
        }

    def counts(self, time: Any, bin_length: float = 100.0) -> xr.DataArray:
        """Active particles per bin at one time (dims bin) or several (dims time, bin)."""
        bins = self.bins(bin_length)
        idx = self.time_indices(time)
        ones = np.ones(self.n_particles)
        data = np.stack([self._mass_per_bin(int(i), bins, ones, None) for i in idx]).astype(np.int64)
        return self._wrap(data, idx, bins, time, "count", "-")

    def concentration(self, time: Any, bin_length: float = 100.0, smoothing: float | str | None = None) -> xr.DataArray:
        """Mass per bin volume (width * depth * bin width) in mass_units m-3; NaN where flow_out is 0.

        The run's ``mass_units`` attribute names the units; a file without it is not a network
        particle file and indexing it raises KeyError rather than quietly labelling the result "kg".

        Args:
            time: an output index, datetime, list of either, slice, or None for all times.
            bin_length: bin size in meters (np.inf for one bin per reach).
            smoothing: None for plain binning, a Gaussian bandwidth in meters, or "auto" for sqrt(2 K dt).
        """
        bins = self.bins(bin_length)
        idx = self.time_indices(time)
        mass = self._ds["mass"].values.astype(np.float64)
        rows = []
        for i in idx:
            m = self._mass_per_bin(int(i), bins, mass, smoothing)
            h = self.provider.hydraulics(self.times[i])
            vol = np.asarray(h["width"])[bins.bin_reach] * np.asarray(h["depth"])[bins.bin_reach] * bins.bin_width
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.where(vol > 0.0, m / vol, np.nan)
            c[np.asarray(h["flow_out"])[bins.bin_reach] <= 0.0] = np.nan
            rows.append(c)
        units = f"{self._ds.attrs['mass_units']} m-3"
        return self._wrap(np.stack(rows), idx, bins, time, "concentration", units)

    def reach_concentration(self, time: Any) -> xr.DataArray:
        """concentration() with one bin per reach (no smoothing: there is only one bin to smooth over)."""
        return self.concentration(time, bin_length=np.inf)

    def _wrap(
        self, data: npt.NDArray[Any], idx: npt.NDArray[np.int64], bins: NetworkBins, time: Any, name: str, units: str
    ) -> xr.DataArray:
        coords = self._bin_coords(bins)
        single = not (time is None or isinstance(time, slice | list | tuple | np.ndarray))
        if single:
            return xr.DataArray(
                data[0],
                dims=("bin",),
                coords=coords,
                name=name,
                attrs={"units": units, "time": str(self.times[idx[0]])},
            )
        coords["time"] = self.times[idx]
        return xr.DataArray(data, dims=("time", "bin"), coords=coords, name=name, attrs={"units": units})

    def persist(
        self, path: str | pathlib.Path, bin_length: float = 100.0, smoothing: float | str | None = None
    ) -> pathlib.Path:
        """Write the full (time, bin) concentration cube to a NetCDF file and return its path."""
        da = self.concentration(None, bin_length=bin_length, smoothing=smoothing)
        da.attrs["bin_length"] = float(bin_length)
        da.to_dataset().to_netcdf(path, engine="h5netcdf")
        return pathlib.Path(path)

    def to_vtp(self, output_dir: str | pathlib.Path, times: Any = None) -> pathlib.Path:
        """Write ``vtp/network_XXXX.vtp`` files and ``network.pvd`` for ParaView; returns the .pvd path.

        Args:
            output_dir: directory to write ``vtp/`` and ``network.pvd`` into (created if missing).
            times: an output index, datetime, list of either, slice, or None for all times.

        Returns:
            Path to the written ``network.pvd`` file.
        """
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        vtp = VTPWriter(out / "vtp")
        pvd = PVDWriter(out / "network.pvd")
        for i in self.time_indices(times):
            df = self.map_positions(int(i))
            scalars = {
                "reach_index": df["reach_index"].to_numpy(),
                "s": df["s"].to_numpy(),
                "mass": df["mass"].to_numpy(),
                "status": df["status"].to_numpy().astype(np.int64),
                "source_index": self._ds["source_index"].values.astype(np.int64),
            }
            f = vtp.write_points(
                df["x"].to_numpy(),
                df["y"].to_numpy(),
                np.zeros(self.n_particles),
                scalars,
                time=float(self.time_seconds[i]),
                tidx=int(i),
                prefix="network",
            )
            if f is not None:
                pvd.add_timestep(float(self.time_seconds[i]), f)
        pvd.write()
        return out / "network.pvd"

    def summary(self) -> str:
        """One-paragraph description of the run."""
        status = self._ds["status"].values[-1]
        exited = int((status == 2).sum())
        return (
            f"NetworkResults: {self.n_particles} particles, {self.n_reach} reaches, {self.times.size} output times "
            f"({self.times[0]} .. {self.times[-1]}), {exited} exited by the end; "
            f"mass units {self._ds.attrs.get('mass_units', '?')}; hydraulics {self._ds.attrs.get('hydraulics_file')}"
        )
