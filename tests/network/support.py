"""Shared builders for the network solver tests: synthetic hydraulics files and an in-memory provider."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr

from fluvial_particle.network.solver import NetworkSolver


G = 9.80665
FIELD_NAMES = ("flow_in", "flow_out", "velocity", "depth", "width", "ustar")
POLYLINE_NAMES = ("vertex_x", "vertex_y", "vertex_dist", "reach_vertex_start", "reach_vertex_count")
UNITS = {
    "reach_id": "-",
    "to_id": "-",
    "to_index": "-",
    "is_outlet": "-",
    "length": "m",
    "slope": "m m-1",
    "mann_n": "s m-1/3",
    "elevation_mid": "m",
    "bankfull_width": "m",
    "bankfull_depth": "m",
    "x_mid": "m",
    "y_mid": "m",
    "flow_in": "m3 s-1",
    "flow_out": "m3 s-1",
    "velocity": "m s-1",
    "depth": "m",
    "width": "m",
    "ustar": "m s-1",
    "residence_time": "s",
    "water_temperature": "degC",
    "vertex_x": "m",
    "vertex_y": "m",
    "vertex_dist": "m",
    "reach_vertex_start": "-",
    "reach_vertex_count": "-",
}
CONVENTIONS_NOTE = (
    "Particle state is (reach index, s) with 0 <= s <= length from the reach's upstream end. "
    "When s exceeds length the particle moves to to_index carrying the unused fraction of the time step; "
    "to_index == -1 is an outlet. Where flow_out is 0 the velocity, depth, width and residence_time are 0; "
    "mask on flow_out > 0."
)


def _static(name: str, values: npt.ArrayLike, dtype: type | None = None) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=dtype), dims=("reach",), attrs={"units": UNITS[name]})


def _field(name: str, values: npt.ArrayLike) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=float), dims=("time", "reach"), attrs={"units": UNITS[name]})


def build_dataset(
    *,
    reach_id: npt.ArrayLike,
    to_index: npt.ArrayLike,
    length: npt.ArrayLike,
    slope: npt.ArrayLike,
    velocity: npt.ArrayLike,
    depth: npt.ArrayLike,
    width: npt.ArrayLike,
    flow_out: npt.ArrayLike,
    flow_in: npt.ArrayLike | None = None,
    times: npt.ArrayLike | None = None,
    polylines: Sequence[npt.ArrayLike] | None = None,
    temperature: npt.ArrayLike | None = None,
) -> xr.Dataset:
    """Build a schema-conforming network hydraulics dataset.

    Time-varying inputs may be (time, reach) arrays or per-reach 1D arrays broadcast over time.
    ``polylines`` is one (n_i, 2) array of x, y vertices per reach, upstream vertex first.
    """
    rid = np.asarray(reach_id, dtype=np.int64)
    n = rid.size
    to_idx = np.asarray(to_index, dtype=np.int32)
    if times is None:
        times = np.datetime64("1979-01-01", "ns") + np.arange(3) * np.timedelta64(1, "D")
    tarr = np.asarray(times, dtype="datetime64[ns]")
    nt = tarr.size

    def tr(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.broadcast_to(np.asarray(a, dtype=float), (nt, n)).copy()

    vel, dep, wid, fout = tr(velocity), tr(depth), tr(width), tr(flow_out)
    fin = fout.copy() if flow_in is None else tr(flow_in)
    slp = np.broadcast_to(np.asarray(slope, dtype=float), (n,)).copy()
    lng = np.asarray(length, dtype=float)
    ustar = np.sqrt(G * dep * np.maximum(slp, 1e-7)[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(fout > 0, wid * dep * lng[None, :] / fout, 0.0)
    to_id = np.where(to_idx >= 0, rid[np.clip(to_idx, 0, n - 1)], 0)

    data: dict[str, xr.DataArray] = {
        "reach_id": _static("reach_id", rid, np.int64),
        "to_id": _static("to_id", to_id, np.int64),
        "to_index": _static("to_index", to_idx, np.int32),
        "is_outlet": _static("is_outlet", (to_idx < 0).astype(np.int8)),
        "length": _static("length", lng),
        "slope": _static("slope", slp),
        "mann_n": _static("mann_n", np.full(n, 0.035)),
        "elevation_mid": _static("elevation_mid", np.linspace(100.0, 10.0, n)),
        "bankfull_width": _static("bankfull_width", wid.max(axis=0)),
        "bankfull_depth": _static("bankfull_depth", dep.max(axis=0)),
        "flow_out": _field("flow_out", fout),
        "flow_in": _field("flow_in", fin),
        "velocity": _field("velocity", vel),
        "depth": _field("depth", dep),
        "width": _field("width", wid),
        "ustar": _field("ustar", ustar),
        "residence_time": _field("residence_time", res),
    }
    if temperature is not None:
        data["water_temperature"] = _field("water_temperature", tr(temperature))
    attrs: dict[str, object] = {
        "title": "test network hydraulics",
        "source_model": "test",
        "n_unconnected": -1,
        "connect_tol": 1.0,
        "crs_wkt": "",
        "conventions_note": CONVENTIONS_NOTE,
    }
    x_mid = np.full(n, np.nan)
    y_mid = np.full(n, np.nan)
    if polylines is not None:
        pls = [np.asarray(p, dtype=float) for p in polylines]
        counts = np.array([len(p) for p in pls], dtype=np.int32)
        starts = (np.cumsum(counts) - counts).astype(np.int64)
        xy = np.concatenate(pls)
        dists = []
        for i, p in enumerate(pls):
            seg = np.diff(p, axis=0)
            d = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
            dists.append(d)
            x_mid[i] = np.interp(d[-1] / 2.0, d, p[:, 0])
            y_mid[i] = np.interp(d[-1] / 2.0, d, p[:, 1])
        dist = np.concatenate(dists)
        data["vertex_x"] = xr.DataArray(xy[:, 0], dims=("vertex",), attrs={"units": "m"})
        data["vertex_y"] = xr.DataArray(xy[:, 1], dims=("vertex",), attrs={"units": "m"})
        data["vertex_dist"] = xr.DataArray(dist, dims=("vertex",), attrs={"units": "m"})
        data["reach_vertex_start"] = _static("reach_vertex_start", starts, np.int64)
        data["reach_vertex_count"] = _static("reach_vertex_count", counts, np.int32)
        attrs["n_unconnected"] = 0
        attrs["crs_wkt"] = 'PROJCRS["test"]'
    data["x_mid"] = _static("x_mid", x_mid)
    data["y_mid"] = _static("y_mid", y_mid)
    ds = xr.Dataset(data, coords={"time": tarr})
    ds.attrs = attrs
    return ds


def three_reach_dataset(**overrides: object) -> xr.Dataset:
    """Two headwaters (reach 0: 1000 m, reach 1: 2000 m) into one outlet (reach 2: 3000 m)."""
    kwargs: dict[str, object] = {
        "reach_id": [101, 102, 103],
        "to_index": [2, 2, -1],
        "length": [1000.0, 2000.0, 3000.0],
        "slope": 0.001,
        "velocity": [1.0, 0.5, 2.0],
        "depth": [1.0, 1.0, 2.0],
        "width": [10.0, 10.0, 20.0],
        "flow_out": [10.0, 5.0, 80.0],
        "polylines": [
            [(-1000.0, 500.0), (0.0, 0.0)],
            [(-2000.0, -500.0), (-1000.0, -250.0), (0.0, 0.0)],
            [(0.0, 0.0), (1500.0, 0.0), (3000.0, 0.0)],
        ],
    }
    kwargs.update(overrides)
    return build_dataset(**kwargs)  # type: ignore[arg-type]


def chain_dataset(
    n_reach: int = 10,
    length: float = 1000.0,
    velocity: float | Sequence[float] = 1.0,
    k_target: float = 10.0,
    width: float = 10.0,
    slope: float = 0.001,
    n_time: int = 4,
) -> xr.Dataset:
    """A uniform chain reach 0 -> 1 -> ... -> n-1 (outlet) whose Fischer coefficient equals ``k_target``.

    Depth is solved from K = 0.011 v^2 w^2 / (d * sqrt(g d S)).
    """
    v = np.broadcast_to(np.asarray(velocity, dtype=float), (n_reach,)).copy()
    depth = (0.011 * v**2 * width**2 / (k_target * np.sqrt(G * slope))) ** (2.0 / 3.0)
    to_index = np.arange(1, n_reach + 1, dtype=np.int32)
    to_index[-1] = -1
    times = np.datetime64("1979-01-01", "ns") + np.arange(n_time) * np.timedelta64(1, "D")
    return build_dataset(
        reach_id=np.arange(1, n_reach + 1),
        to_index=to_index,
        length=np.full(n_reach, length),
        slope=slope,
        velocity=v,
        depth=depth,
        width=np.full(n_reach, width),
        flow_out=v * depth * width,
        times=times,
    )


def write_network_file(path: pathlib.Path, ds: xr.Dataset) -> pathlib.Path:
    """Write ``ds`` as NetCDF4 through h5netcdf and return the path."""
    ds.to_netcdf(path, engine="h5netcdf")
    return pathlib.Path(path)


class ArrayHydraulicsProvider:
    """In-memory HydraulicsProvider with hold-in-time semantics, for tests and as the BMI-path prototype."""

    def __init__(
        self,
        times: npt.ArrayLike,
        static: dict[str, npt.NDArray[np.generic]],
        fields: dict[str, npt.ArrayLike],
        *,
        dtype: npt.DTypeLike = np.float64,
    ) -> None:
        self.times = np.asarray(times, dtype="datetime64[ns]")
        self.static = dict(static)
        self.dtype = np.dtype(dtype)
        self._fields = {k: np.asarray(v, dtype=self.dtype) for k, v in fields.items()}
        self.crs_wkt = ""
        self.n_reach = int(self.static["reach_id"].size)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, *, dtype: npt.DTypeLike = np.float64) -> ArrayHydraulicsProvider:
        static = {str(v): ds[v].values for v in ds.data_vars if ds[v].dims in {("reach",), ("vertex",)}}
        names = FIELD_NAMES + (("water_temperature",) if "water_temperature" in ds else ())
        fields = {n: ds[n].values for n in names}
        return cls(ds["time"].values, static, fields, dtype=dtype)

    def hydraulics(self, t: np.datetime64) -> dict[str, npt.NDArray[np.floating]]:
        """Fields of the last timestamp not after ``t``."""
        tt = np.datetime64(t, "ns")
        k = int(np.searchsorted(self.times, tt, side="right")) - 1
        if k < 0 or tt > self.times[-1]:
            raise ValueError(f"time {tt} outside provider range {self.times[0]}..{self.times[-1]}")
        return {name: a[k].copy() for name, a in self._fields.items()}

    def close(self) -> None:
        """Nothing to release."""


class HalfSpeed(NetworkSolver):
    """Test model: records releases and advects at half the reach velocity (no parameters)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.released: list[npt.NDArray[np.int64]] = []
        self.behave_calls: list[tuple[float, float, float]] = []

    def on_release(self, idx: npt.NDArray[np.int64], h: object) -> None:  # noqa: ARG002
        self.released.append(idx.copy())

    def behave(self, h: object, tau: npt.NDArray[np.float64], t: float, dt: float) -> npt.NDArray[np.float64]:  # noqa: ARG002
        self.behave_calls.append((t, dt, float(tau[0])))
        return np.full(self.n, 0.5)


class NotASolver:
    """Registry test: a class that does not subclass NetworkSolver."""
