"""Hydraulics providers: the Protocol the solver consumes and the file-backed streaming implementation."""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import h5py
import numpy as np
import numpy.typing as npt
import xarray as xr

from .network import Network


REQUIRED_STATIC: dict[str, str | None] = {
    "reach_id": None,
    "to_index": None,
    "is_outlet": None,
    "length": "m",
    "slope": "m m-1",
}
OPTIONAL_STATIC: tuple[str, ...] = ("mann_n", "elevation_mid", "bankfull_width", "bankfull_depth", "x_mid", "y_mid")
REQUIRED_FIELDS: dict[str, str] = {
    "flow_in": "m3 s-1",
    "flow_out": "m3 s-1",
    "velocity": "m s-1",
    "depth": "m",
    "width": "m",
    "ustar": "m s-1",
}
OPTIONAL_FIELDS: tuple[str, ...] = ("water_temperature",)
POLYLINE_VARS: tuple[str, ...] = ("vertex_x", "vertex_y", "vertex_dist", "reach_vertex_start", "reach_vertex_count")
INTERPOLATIONS = ("linear", "hold")
MAX_TIME_CHUNK = 32

FloatArray = npt.NDArray[np.floating[Any]]


@runtime_checkable
class HydraulicsProvider(Protocol):
    """Per-step hydraulics for the network solver; dict keys are the export's variable names."""

    times: npt.NDArray[np.datetime64]
    static: Mapping[str, npt.NDArray[Any]]
    dtype: np.dtype[Any]
    crs_wkt: str
    n_reach: int

    def hydraulics(self, t: np.datetime64) -> dict[str, FloatArray]:
        """Per-reach velocity, depth, width, ustar, flow_in, flow_out (and water_temperature) at time t."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class StaticArrays(Mapping[str, npt.NDArray[Any]]):
    """Read-only mapping of static arrays whose polyline block is loaded on first access."""

    def __init__(self, data: dict[str, npt.NDArray[Any]], lazy_keys: Sequence[str], loader: Any) -> None:
        """Wrap eagerly-loaded static arrays plus a deferred polyline block.

        Args:
            data: eagerly-loaded static arrays, keyed by variable name.
            lazy_keys: names of polyline variables not yet present in ``data``.
            loader: zero-argument callable returning the polyline block as a dict.
        """
        self._data = data
        self._lazy_keys = tuple(lazy_keys)
        self._loader = loader
        self.polylines_loaded = not self._lazy_keys

    def __getitem__(self, key: str) -> npt.NDArray[Any]:
        """Return the array for ``key``, loading the polyline block on first access to a lazy key."""
        if key not in self._data and key in self._lazy_keys and not self.polylines_loaded:
            self._data.update(self._loader())
            self.polylines_loaded = True
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over all keys, loaded and not-yet-loaded."""
        return iter({**dict.fromkeys(self._data), **dict.fromkeys(self._lazy_keys)})

    def __len__(self) -> int:
        """Return the number of keys, loaded and not-yet-loaded."""
        return len(set(self._data) | set(self._lazy_keys))

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is a loaded or not-yet-loaded array name."""
        return key in self._data or key in self._lazy_keys


def _walk_to_outlets(to_index: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """True for reaches whose downstream walk reaches an outlet; False marks a cycle."""
    n = to_index.size
    cur = to_index.copy()
    done = cur < 0
    for _ in range(n):
        if done.all():
            break
        cur = np.where(done, -1, to_index[np.clip(cur, 0, n - 1)])
        done |= cur < 0
    return done


def subset_static(static: dict[str, npt.NDArray[Any]], sel: npt.NDArray[np.int64]) -> dict[str, npt.NDArray[Any]]:
    """Restrict per-reach static arrays (not the polyline block) to indices ``sel`` and remap to_index."""
    out = {k: v[sel] for k, v in static.items() if k not in POLYLINE_VARS}
    n = static["reach_id"].size
    new_index = np.full(n, -1, dtype=np.int64)
    new_index[sel] = np.arange(sel.size)
    old_to = np.asarray(static["to_index"], dtype=np.int64)[sel]
    new_to = np.where(old_to >= 0, new_index[np.clip(old_to, 0, n - 1)], -1)
    out["to_index"] = new_to.astype(np.int32)
    out["is_outlet"] = (new_to < 0).astype(np.int8)
    return out


class FileHydraulicsProvider:
    """Streaming provider over a network hydraulics NetCDF export.

    Args:
        path: the export file.
        interpolation: "linear" between timestamps or "hold" the last timestamp.
        dtype: float dtype for the time-varying fields.
        reach_subset: sequence of reach ids, or {"outlet": reach_id} for the upstream closure of that reach.
        time_window: (start, end) datetime64 bounds accepted by hydraulics(); default the file's range.

    Raises:
        ValueError: schema, units, topology, or time-axis problems (message names the variable).
        KeyError: a reach id in reach_subset is not in the file.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        interpolation: str = "linear",
        dtype: npt.DTypeLike = "float64",
        reach_subset: Sequence[int] | Mapping[str, int] | None = None,
        time_window: tuple[np.datetime64, np.datetime64] | None = None,
    ) -> None:
        """Open, validate, and index a network hydraulics NetCDF export.

        Args:
            path: the export file.
            interpolation: "linear" between timestamps or "hold" the last timestamp.
            dtype: float dtype for the time-varying fields.
            reach_subset: sequence of reach ids, or {"outlet": reach_id} for the upstream closure of
                that reach.
            time_window: (start, end) datetime64 bounds accepted by hydraulics(); default the file's range.

        Raises:
            ValueError: schema, units, topology, or time-axis problems (message names the variable).
        """
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")
        self.path = pathlib.Path(path)
        self.interpolation = interpolation
        self.dtype = np.dtype(dtype)
        self._ds = xr.open_dataset(self.path, engine="h5netcdf")
        self._validate_schema()
        self.times: npt.NDArray[np.datetime64] = self._ds["time"].values.astype("datetime64[ns]")
        self._n_time = int(self.times.size)
        if self._n_time > 1 and not np.all(np.diff(self.times) > np.timedelta64(0, "ns")):
            raise ValueError("time must be strictly increasing")
        self.crs_wkt = str(self._ds.attrs.get("crs_wkt", ""))
        self.conventions_note = str(self._ds.attrs.get("conventions_note", ""))
        self.has_temperature = "water_temperature" in self._ds
        self._field_names = tuple(REQUIRED_FIELDS) + (("water_temperature",) if self.has_temperature else ())
        full_static = self._load_static()
        self._validate_topology(full_static)
        self._n_reach_file = int(full_static["reach_id"].size)
        self.subset_index: npt.NDArray[np.int64] | None = self._resolve_subset(full_static, reach_subset)
        data = full_static if self.subset_index is None else subset_static(full_static, self.subset_index)
        lazy = POLYLINE_VARS if all(v in self._ds for v in POLYLINE_VARS) else ()
        self.static: StaticArrays = StaticArrays(data, lazy, self._load_polylines)
        self.n_reach = int(data["reach_id"].size)
        self._window_k: int = -2
        self._slices: dict[int, dict[str, FloatArray]] = {}
        self._time_window: tuple[np.datetime64, np.datetime64] | None = None
        self.time_window = time_window
        self._check_chunking()

    # ---- context manager -------------------------------------------------
    def __enter__(self) -> FileHydraulicsProvider:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the underlying dataset on context exit."""
        self.close()

    def close(self) -> None:
        """Close the underlying dataset."""
        self._ds.close()

    # ---- validation --------------------------------------------------------
    def _validate_schema(self) -> None:
        """Check required variables, dimensions, and units are present and consistent.

        Raises:
            ValueError: a required variable is missing, has the wrong dimensions, has the wrong
                units, or time does not decode to datetime64.
        """
        ds = self._ds
        missing = [v for v in list(REQUIRED_STATIC) + list(REQUIRED_FIELDS) if v not in ds]
        if missing:
            raise ValueError(f"hydraulics file is missing required variables: {missing}")
        for name in REQUIRED_STATIC:
            if ds[name].dims != ("reach",):
                raise ValueError(f"{name} must have dimensions ('reach',), got {ds[name].dims}")
        for name in REQUIRED_FIELDS:
            if ds[name].dims != ("time", "reach"):
                raise ValueError(f"{name} must have dimensions ('time', 'reach'), got {ds[name].dims}")
        expected = {**{k: v for k, v in REQUIRED_STATIC.items() if v}, **REQUIRED_FIELDS}
        for name, units in expected.items():
            got = ds[name].attrs.get("units")
            if got is None:
                warnings.warn(f"{name} has no units attribute; assuming {units!r}", UserWarning, stacklevel=3)
            elif got != units:
                raise ValueError(f"{name} units are {got!r}, expected {units!r}")
        if ds["time"].dtype.kind != "M":
            raise ValueError("time must decode to datetime64")

    def _load_static(self) -> dict[str, npt.NDArray[Any]]:
        """Eagerly load the per-reach static arrays (not the polyline block)."""
        out: dict[str, npt.NDArray[Any]] = {}
        for name in list(REQUIRED_STATIC) + [v for v in OPTIONAL_STATIC if v in self._ds]:
            arr = self._ds[name].values
            if name in {"reach_id"}:
                arr = arr.astype(np.int64)
            elif name == "to_index":
                if arr.dtype.kind not in "iu":
                    raise ValueError("to_index must be an integer array")
                arr = arr.astype(np.int32)
            elif name == "is_outlet":
                arr = arr.astype(np.int8)
            else:
                arr = arr.astype(np.float64)
            out[name] = arr
        return out

    def _validate_topology(self, static: dict[str, npt.NDArray[Any]]) -> None:
        """Check to_index is in range, free of cycles, and consistent with is_outlet.

        Raises:
            ValueError: to_index is out of range, contains a cycle, or disagrees with is_outlet.
        """
        to_index = np.asarray(static["to_index"], dtype=np.int64)
        n = to_index.size
        bad = np.nonzero((to_index < -1) | (to_index >= n))[0]
        if bad.size:
            raise ValueError(f"to_index out of range [-1, {n}) at reach indices {bad[:10].tolist()}")
        done = _walk_to_outlets(to_index)
        if not done.all():
            chain = np.nonzero(~done)[0]
            raise ValueError(f"to_index contains a cycle through reach indices {chain[:10].tolist()}")
        is_outlet = np.asarray(static["is_outlet"]) != 0
        if not np.array_equal(is_outlet, to_index < 0):
            raise ValueError("is_outlet disagrees with to_index == -1")

    def _validate_polylines(
        self, start: npt.NDArray[np.int64], count: npt.NDArray[np.int64], vdist: FloatArray
    ) -> None:
        """Check the polyline block's index arrays are in range and vertex_dist is monotone per reach.

        Raises:
            ValueError: reach_vertex_start/reach_vertex_count exceed the vertex dimension, or
                vertex_dist decreases within a reach.
        """
        if np.any(start < 0) or np.any(count < 0) or np.any(start + count > vdist.size):
            raise ValueError("reach_vertex_start + reach_vertex_count exceeds the vertex dimension")
        idx = np.repeat(start, count) + (np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count))
        reach_of = np.repeat(np.arange(start.size), count)
        d = vdist[idx]
        same = reach_of[1:] == reach_of[:-1]
        if np.any((np.diff(d) < 0.0) & same):
            raise ValueError("vertex_dist must be non-decreasing within each reach")

    # ---- subsetting --------------------------------------------------------
    def _resolve_subset(
        self, static: dict[str, npt.NDArray[Any]], reach_subset: Sequence[int] | Mapping[str, int] | None
    ) -> npt.NDArray[np.int64] | None:
        """Resolve a reach_subset argument to sorted reach indices, or None for no subsetting.

        Raises:
            ValueError: reach_subset is a mapping without exactly the key "outlet".
        """
        if reach_subset is None:
            return None
        network = Network(static)
        if isinstance(reach_subset, Mapping):
            if set(reach_subset) != {"outlet"}:
                raise ValueError("reach_subset mapping must be {'outlet': reach_id}")
            ids = network.upstream_of(int(reach_subset["outlet"]))
        else:
            ids = np.asarray(list(reach_subset), dtype=np.int64)
        sel = np.unique(network.index_of_many(ids))  # sorted: keeps file order
        return sel.astype(np.int64)

    def _load_polylines(self) -> dict[str, npt.NDArray[Any]]:
        """Load, validate, and (if subsetting) restrict the polyline block."""
        start = self._ds["reach_vertex_start"].values.astype(np.int64)
        count = self._ds["reach_vertex_count"].values.astype(np.int64)
        vx = self._ds["vertex_x"].values.astype(np.float64)
        vy = self._ds["vertex_y"].values.astype(np.float64)
        vd = self._ds["vertex_dist"].values.astype(np.float64)
        self._validate_polylines(start, count, vd)
        if self.subset_index is not None:
            start, count = start[self.subset_index], count[self.subset_index]
            idx = np.repeat(start, count) + (np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count))
            vx, vy, vd = vx[idx], vy[idx], vd[idx]
            start = (np.cumsum(count) - count).astype(np.int64)
        return {
            "vertex_x": vx,
            "vertex_y": vy,
            "vertex_dist": vd,
            "reach_vertex_start": start,
            "reach_vertex_count": count.astype(np.int32),
        }

    # ---- housekeeping ------------------------------------------------------
    def _check_chunking(self) -> None:
        """Warn when the velocity variable's HDF5 chunking will make streaming slow."""
        with h5py.File(self.path, "r") as f:
            chunks = f["velocity"].chunks
        if chunks is None:
            return
        if chunks[1] < self._n_reach_file or chunks[0] > MAX_TIME_CHUNK:
            warnings.warn(
                f"velocity is chunked {chunks}; a time slice spans several chunks or the time chunk exceeds "
                f"{MAX_TIME_CHUNK}, so streaming will be slow. Rechunk to (small, {self._n_reach_file}).",
                UserWarning,
                stacklevel=3,
            )

    @property
    def time_window(self) -> tuple[np.datetime64, np.datetime64] | None:
        """Bounds accepted by hydraulics(); None means the file's full range."""
        return self._time_window

    @time_window.setter
    def time_window(self, value: tuple[np.datetime64, np.datetime64] | None) -> None:
        if value is None:
            self._time_window = None
            return
        t0, t1 = np.asarray(value, dtype="datetime64[ns]")
        if t0 < self.times[0] or t1 > self.times[-1] or t0 > t1:
            raise ValueError(f"time_window {value} is outside the file range {self.times[0]}..{self.times[-1]}")
        self._time_window = (t0, t1)

    def memory_estimate(self, n_particles: int) -> dict[str, int]:
        """Bytes for the two-slice window, static arrays, and particle arrays for ``n_particles``."""
        n_fields = len(self._field_names)
        window = 2 * self.n_reach * n_fields * self.dtype.itemsize
        static = sum(int(v.nbytes) for k, v in self.static._data.items())
        # reach(int32) + s(dtype) + prev_reach(int32) + status(int8) + mass(f64) + release(3 arrays) + exit(2 arrays)
        per_particle = 4 + self.dtype.itemsize + 4 + 1 + 8 + (4 + 8 + 8) + (8 + 4)
        return {
            "window_bytes": int(window),
            "static_bytes": int(static),
            "particle_bytes": int(per_particle * n_particles),
            "output_time_bytes": int((4 + self.dtype.itemsize + 1) * n_particles),
        }

    def hydraulics(self, t: np.datetime64) -> dict[str, FloatArray]:
        """Implemented in the streaming task."""
        raise NotImplementedError
