# 1D River-Network Particle Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `fluvial_particle.network`, a 1D river-network passive particle tracker that reads the pywatershed network hydraulics NetCDF export, transports particles by exact advection plus Fischer dispersion, writes a NetCDF particle file, and post-processes arrival times, map positions, and binned concentration.

**Architecture:** A new subpackage with one module per responsibility: a streaming file-backed hydraulics provider behind a small Protocol, a static `Network` (topology, polylines, bins), mass-loading sources expanded to a per-particle schedule, a vectorized solver whose step is exact advection with time carry followed by one dispersive kick with displacement carry, an h5netcdf writer (serial or mpio), and a lazy `NetworkResults` reader that derives everything else. Nothing in the existing VTK 2D/3D path changes except a generic `write_points` method on `VTPWriter`.

**Tech Stack:** Python 3.10+, numpy, xarray (engine h5netcdf), h5netcdf, h5py, pandas, scipy (tests only), matplotlib (notebook only), mpi4py (optional), pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-13-network-particle-solver-design.md`

## Global Constraints

- All commands run in the conda env: `conda run -n fluvial-particle <cmd>`. Never `uv sync`; add deps with `conda run -n fluvial-particle uv pip install -e ".[dev]"`.
- Python floor `>=3.10`; use `from __future__ import annotations` and `X | None` unions; TOML via `tomllib` with the `tomli` fallback already declared in `pyproject.toml`.
- mypy is strict for `src/` (`disallow_untyped_defs`, `disallow_any_generics`, `warn_return_any`): every function in `src/fluvial_particle/network/` is fully annotated; use `npt.NDArray[np.float64]` style (`import numpy.typing as npt`). `ignore_missing_imports = true` covers xarray/h5netcdf/mpi4py.
- ruff: google docstring convention, line length 120, `D` rules on `src/` (public modules, classes, functions all need docstrings with `Args:`/`Returns:`/`Raises:` where applicable), `D`/`S101`/`PLR2004` ignored in `tests/**`. Run `ruff check` and `ruff format` before every commit.
- Variable names in provider dicts and the output file are the export's names exactly: `flow_in`, `flow_out`, `velocity`, `depth`, `width`, `ustar`, `water_temperature`, `reach_id`, `to_index`, `is_outlet`, `length`, `slope`, `vertex_x`, `vertex_y`, `vertex_dist`, `reach_vertex_start`, `reach_vertex_count`.
- Units strings the provider validates: `length` "m", `slope` "m m-1", `flow_in`/`flow_out` "m3 s-1", `velocity` "m s-1", `depth` "m", `width` "m", `ustar` "m s-1".
- Particle state convention: `(reach index, s)` with `0 <= s <= length[reach]` from the reach's upstream end; `to_index == -1` is an outlet; `status` 0 unreleased, 1 active, 2 exited.
- Dispersion: `K = scale * 0.011 * v^2 * w^2 / (d * ustar)`, 0 where `depth`, `ustar`, or `velocity` is 0; per-step random variance is exactly `2 K tau`.
- Output file name `network_particles.nc`; `(time, particle)` variables chunked `(1, min(N, 2**18))`; `time` dimension unlimited.
- The DRB file is never committed. Tests build synthetic files with `tests/network/support.py`. The DRB path for the optional test comes from `FLUVIAL_PARTICLE_DRB_FILE`.
- Commit after every task with the attribution line: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Branch: work on `feature/network-solver` cut from `feature/network-solver-spec` (which holds the spec commit).

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/fluvial_particle/network/__init__.py` | Public exports of the subpackage |
| `src/fluvial_particle/network/dispersion.py` | `fischer_coefficient`, `dispersion_coefficient` |
| `src/fluvial_particle/network/network.py` | `Network` (topology, ids, parents, headwaters, outlets, upstream closure, `map_position`), `NetworkBins` |
| `src/fluvial_particle/network/provider.py` | `HydraulicsProvider` Protocol, `FileHydraulicsProvider` (validate, subset, stream, interpolate), `subset_static` helper |
| `src/fluvial_particle/network/config.py` | `DispersionConfig`, `NetworkConfig`, `get_network_config_template` |
| `src/fluvial_particle/network/sources.py` | `ParticleSchedule`, `expand_sources`, `estimate_particles`, time parsing |
| `src/fluvial_particle/network/solver.py` | `NetworkSolver` with `step()` |
| `src/fluvial_particle/network/writer.py` | `NetworkWriter` (h5netcdf, serial or mpio) |
| `src/fluvial_particle/network/run.py` | `run_network_simulation`, `diagnostics_report` |
| `src/fluvial_particle/network/results.py` | `NetworkResults` |
| `src/fluvial_particle/cli.py` | add `network_serial`, `network_mpi` |
| `src/fluvial_particle/io/vtp_writer.py` | add `VTPWriter.write_points` |
| `src/fluvial_particle/__init__.py` | re-export network API |
| `pyproject.toml` | deps, dev extras, console scripts, pytest markers |
| `tests/network/__init__.py`, `tests/network/support.py` | dataset builders, `ArrayHydraulicsProvider` |
| `tests/network/test_*.py` | one test module per source module, plus `test_analytical.py`, `test_run.py`, `test_cli.py`, `test_drb.py` |
| `docs/network.rst`, `docs/optionsfile.rst`, `docs/reference.rst`, `docs/index.rst`, `docs/output.rst` | documentation |
| `.claude/rules/network.md` | rules page for the subpackage |
| `notebooks/network-drb-demo.ipynb` | demo |

---

### Task 1: Packaging, test support module, and branch

**Files:**
- Modify: `pyproject.toml` (dependencies, dev extras, scripts, pytest markers)
- Create: `src/fluvial_particle/network/__init__.py`
- Create: `tests/network/__init__.py`
- Create: `tests/network/support.py`
- Test: `tests/network/test_support.py`

**Interfaces:**
- Produces: `tests.network.support.build_dataset(...) -> xr.Dataset`, `three_reach_dataset(**overrides) -> xr.Dataset`, `chain_dataset(n_reach, length, velocity, k_target, ...) -> xr.Dataset`, `write_network_file(path, ds) -> pathlib.Path`, `ArrayHydraulicsProvider(times, static, fields, dtype=...)` with `.times`, `.static`, `.dtype`, `.crs_wkt`, `.n_reach`, `.hydraulics(t)`, `.close()`, and classmethod `from_dataset(ds, dtype=...)`.

- [ ] **Step 1: Create the working branch**

```bash
git checkout feature/network-solver-spec
git checkout -b feature/network-solver
```

- [ ] **Step 2: Add dependencies, scripts, and pytest markers to `pyproject.toml`**

In `[project] dependencies` replace the list with:

```toml
dependencies = [
    "tomli>=2.0.0; python_version < '3.11'",  # TOML parser backport for Python 3.10
    "xarray>=2023.1",   # network hydraulics input and results
    "h5netcdf>=1.1",    # NetCDF4 output on h5py (serial or mpio)
]
```

In `[project.optional-dependencies] dev`, after `"pandas>=2.0.0",` add:

```toml
    "scipy>=1.10",        # analytical acceptance tests (inverse Gaussian, KS)
    "matplotlib>=3.7",    # demo notebook
```

In `[project.scripts]` add:

```toml
fluvial_particle_network = "fluvial_particle.cli:network_serial"
fluvial_particle_network_mpi = "fluvial_particle.cli:network_mpi"
```

Add a new table after `[tool.ruff.format]`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: analytical acceptance tests that take tens of seconds",
]
```

Then reinstall: `conda run -n fluvial-particle uv pip install -e ".[dev]"`.

- [ ] **Step 3: Create the empty subpackage and test package**

`src/fluvial_particle/network/__init__.py`:

```python
"""1D river-network particle tracking driven by network hydraulics exports."""
```

`tests/network/__init__.py`: empty file.

- [ ] **Step 4: Write the failing support test**

`tests/network/test_support.py`:

```python
"""Tests for the network test-support builders."""

import numpy as np
import xarray as xr

from tests.network.support import ArrayHydraulicsProvider, chain_dataset, three_reach_dataset, write_network_file


def test_three_reach_dataset_schema():
    ds = three_reach_dataset()
    assert ds.sizes == {"reach": 3, "time": 3, "vertex": 8}
    for name in ("flow_in", "flow_out", "velocity", "depth", "width", "ustar"):
        assert ds[name].dims == ("time", "reach")
        assert "units" in ds[name].attrs
    assert list(ds["to_index"].values) == [2, 2, -1]
    assert list(ds["is_outlet"].values) == [0, 0, 1]
    assert list(ds["reach_vertex_start"].values) == [0, 2, 5]
    assert list(ds["reach_vertex_count"].values) == [2, 3, 3]
    np.testing.assert_allclose(ds["vertex_dist"].values[5:8], [0.0, 1500.0, 3000.0])
    np.testing.assert_allclose(ds["ustar"].values[0], np.sqrt(9.80665 * np.array([1.0, 1.0, 2.0]) * 0.001))


def test_write_and_reopen(tmp_path):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset(temperature=5.0))
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds["time"].dtype.kind == "M"
        assert "water_temperature" in ds
        assert ds.attrs["conventions_note"]


def test_chain_dataset_hits_target_dispersion():
    ds = chain_dataset(n_reach=4, length=500.0, velocity=1.0, k_target=10.0)
    v, d, w, u = (ds[n].values[0] for n in ("velocity", "depth", "width", "ustar"))
    np.testing.assert_allclose(0.011 * v**2 * w**2 / (d * u), 10.0)
    assert list(ds["to_index"].values) == [1, 2, 3, -1]


def test_array_provider_hold_semantics():
    ds = three_reach_dataset(velocity=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    t = np.datetime64("1979-01-02T12:00", "ns")
    assert prov.hydraulics(t)["velocity"][0] == 2.0
    assert prov.n_reach == 3
    assert "reach_vertex_start" in prov.static
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `conda run -n fluvial-particle pytest tests/network/test_support.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.network.support'`

- [ ] **Step 6: Write `tests/network/support.py`**

```python
"""Shared builders for the network solver tests: synthetic hydraulics files and an in-memory provider."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr


G = 9.80665
FIELD_NAMES = ("flow_in", "flow_out", "velocity", "depth", "width", "ustar")
POLYLINE_NAMES = ("vertex_x", "vertex_y", "vertex_dist", "reach_vertex_start", "reach_vertex_count")
UNITS = {
    "reach_id": "-", "to_id": "-", "to_index": "-", "is_outlet": "-",
    "length": "m", "slope": "m m-1", "mann_n": "s m-1/3", "elevation_mid": "m",
    "bankfull_width": "m", "bankfull_depth": "m", "x_mid": "m", "y_mid": "m",
    "flow_in": "m3 s-1", "flow_out": "m3 s-1", "velocity": "m s-1", "depth": "m",
    "width": "m", "ustar": "m s-1", "residence_time": "s", "water_temperature": "degC",
    "vertex_x": "m", "vertex_y": "m", "vertex_dist": "m",
    "reach_vertex_start": "-", "reach_vertex_count": "-",
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
        static = {str(v): ds[v].values for v in ds.data_vars if ds[v].dims in (("reach",), ("vertex",))}
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
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `conda run -n fluvial-particle pytest tests/network/test_support.py -v`
Expected: 4 passed

- [ ] **Step 8: Lint and commit**

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
git add pyproject.toml src/fluvial_particle/network/__init__.py tests/network
git commit -m "Add network subpackage scaffold, dependencies, and test support builders

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Dispersion coefficient

**Files:**
- Create: `src/fluvial_particle/network/dispersion.py`
- Test: `tests/network/test_dispersion.py`

**Interfaces:**
- Produces: `fischer_coefficient(velocity, depth, width, ustar, *, scale=1.0, cap=None) -> NDArray[float64]`; `dispersion_coefficient(fields: Mapping[str, NDArray], model: str, *, scale=1.0, cap=None, value=None) -> NDArray[float64]` where `model` is `"fischer" | "constant" | "none"`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_dispersion.py`:

```python
"""Tests for the longitudinal dispersion coefficient."""

import numpy as np
import pytest

from fluvial_particle.network.dispersion import dispersion_coefficient, fischer_coefficient


def test_fischer_value():
    k = fischer_coefficient(np.array([1.0]), np.array([2.0]), np.array([10.0]), np.array([0.1]))
    np.testing.assert_allclose(k, 0.011 * 1.0 * 100.0 / (2.0 * 0.1))


def test_fischer_zero_where_dry_or_still():
    v = np.array([1.0, 0.0, 1.0, 1.0])
    d = np.array([1.0, 1.0, 0.0, 1.0])
    w = np.array([10.0, 10.0, 10.0, 10.0])
    u = np.array([0.1, 0.1, 0.1, 0.0])
    k = fischer_coefficient(v, d, w, u)
    assert k[0] > 0
    assert list(k[1:]) == [0.0, 0.0, 0.0]


def test_fischer_scale_and_cap():
    v, d, w, u = (np.array([x]) for x in (1.0, 2.0, 10.0, 0.1))
    base = fischer_coefficient(v, d, w, u)[0]
    assert fischer_coefficient(v, d, w, u, scale=2.0)[0] == pytest.approx(2 * base)
    assert fischer_coefficient(v, d, w, u, cap=1.0)[0] == 1.0


def test_dispersion_models():
    fields = {
        "velocity": np.array([1.0, 1.0]),
        "depth": np.array([2.0, 2.0]),
        "width": np.array([10.0, 10.0]),
        "ustar": np.array([0.1, 0.1]),
        "flow_out": np.array([5.0, 0.0]),
    }
    assert list(dispersion_coefficient(fields, "none")) == [0.0, 0.0]
    k = dispersion_coefficient(fields, "constant", value=3.0)
    assert list(k) == [3.0, 0.0]
    k = dispersion_coefficient(fields, "fischer")
    assert k[0] == pytest.approx(5.5)
    with pytest.raises(ValueError, match="model"):
        dispersion_coefficient(fields, "bogus")
    with pytest.raises(ValueError, match="value"):
        dispersion_coefficient(fields, "constant")
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_dispersion.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `dispersion.py`**

```python
"""Longitudinal dispersion coefficients for 1D river-network transport."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


FISCHER_CONSTANT = 0.011
"""Fischer et al. (1979) coefficient in K = 0.011 v^2 w^2 / (d u*)."""

DISPERSION_MODELS = ("fischer", "constant", "none")


def fischer_coefficient(
    velocity: npt.ArrayLike,
    depth: npt.ArrayLike,
    width: npt.ArrayLike,
    ustar: npt.ArrayLike,
    *,
    scale: float = 1.0,
    cap: float | None = None,
) -> npt.NDArray[np.float64]:
    """Fischer longitudinal dispersion coefficient K = scale * 0.011 v^2 w^2 / (d u*).

    Args:
        velocity: reach velocity (m/s).
        depth: flow depth (m).
        width: flow width (m).
        ustar: shear velocity (m/s).
        scale: multiplier on the Fischer value.
        cap: optional upper bound (m^2/s).

    Returns:
        K (m^2/s), 0 where depth, ustar, or velocity is 0.
    """
    v = np.asarray(velocity, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    w = np.asarray(width, dtype=np.float64)
    u = np.asarray(ustar, dtype=np.float64)
    denom = d * u
    ok = (denom > 0.0) & (v != 0.0)
    k = np.zeros(np.broadcast(v, d, w, u).shape, dtype=np.float64)
    k[ok] = scale * FISCHER_CONSTANT * v[ok] ** 2 * w[ok] ** 2 / denom[ok]
    if cap is not None:
        np.minimum(k, cap, out=k)
    return k


def dispersion_coefficient(
    fields: Mapping[str, npt.NDArray[np.floating]],
    model: str,
    *,
    scale: float = 1.0,
    cap: float | None = None,
    value: float | None = None,
) -> npt.NDArray[np.float64]:
    """Per-reach dispersion coefficient for the configured model.

    Args:
        fields: per-reach hydraulics with keys velocity, depth, width, ustar, flow_out.
        model: "fischer", "constant", or "none".
        scale: Fischer multiplier.
        cap: Fischer upper bound (m^2/s).
        value: constant K (m^2/s) for the "constant" model.

    Returns:
        K per reach (m^2/s); 0 where flow_out is 0 for every model.

    Raises:
        ValueError: unknown model, or "constant" without a value.
    """
    n = np.asarray(fields["velocity"]).shape[0]
    if model == "none":
        return np.zeros(n, dtype=np.float64)
    if model == "constant":
        if value is None:
            raise ValueError("dispersion model 'constant' requires a value")
        k = np.full(n, float(value), dtype=np.float64)
        k[np.asarray(fields["flow_out"]) <= 0.0] = 0.0
        return k
    if model == "fischer":
        return fischer_coefficient(fields["velocity"], fields["depth"], fields["width"], fields["ustar"], scale=scale, cap=cap)
    raise ValueError(f"unknown dispersion model {model!r}; expected one of {DISPERSION_MODELS}")
```

- [ ] **Step 4: Run to verify pass, lint, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_dispersion.py -v` → 4 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/dispersion.py tests/network/test_dispersion.py
git commit -m "Add Fischer and constant dispersion coefficients for the network solver

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Network topology and map positions

**Files:**
- Create: `src/fluvial_particle/network/network.py`
- Test: `tests/network/test_network.py`

**Interfaces:**
- Consumes: provider `static` mapping (from `ArrayHydraulicsProvider.from_dataset(ds).static` in tests).
- Produces: `Network(static, crs_wkt="")` with `n_reach: int`, `reach_id: NDArray[int64]`, `to_index: NDArray[int32]`, `length: NDArray[float64]`, `is_outlet: NDArray[bool]`, `has_polylines: bool`, `crs_wkt: str`, `index_of(reach_id: int) -> int`, `index_of_many(ids) -> NDArray[int64]`, `id_of(index: int) -> int`, `parents(index: int) -> NDArray[int64]`, `headwaters(as_index=False) -> NDArray`, `outlets(as_index=False) -> NDArray`, `upstream_of(reach_id: int) -> NDArray[int64]` (ids, inclusive), `map_position(reach, s) -> tuple[NDArray[float64], NDArray[float64]]`, `polylines() -> list[tuple[NDArray, NDArray]]`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_network.py`:

```python
"""Tests for Network topology and polyline mapping."""

import numpy as np
import pytest

from fluvial_particle.network.network import Network
from tests.network.support import ArrayHydraulicsProvider, chain_dataset, three_reach_dataset


@pytest.fixture
def net():
    return Network(ArrayHydraulicsProvider.from_dataset(three_reach_dataset()).static, crs_wkt="PROJCRS")


def test_basic_attributes(net):
    assert net.n_reach == 3
    assert list(net.reach_id) == [101, 102, 103]
    assert list(net.is_outlet) == [False, False, True]
    assert net.index_of(102) == 1
    assert net.id_of(2) == 103
    assert list(net.index_of_many([103, 101])) == [2, 0]
    with pytest.raises(KeyError):
        net.index_of(999)
    assert net.crs_wkt == "PROJCRS"


def test_topology_queries(net):
    assert list(net.headwaters()) == [101, 102]
    assert list(net.headwaters(as_index=True)) == [0, 1]
    assert list(net.outlets()) == [103]
    assert sorted(net.parents(2)) == [0, 1]
    assert net.parents(0).size == 0
    assert sorted(net.upstream_of(103)) == [101, 102, 103]
    assert list(net.upstream_of(101)) == [101]


def test_map_position_hand_values(net):
    x, y = net.map_position(np.array([2, 2, 2]), np.array([0.0, 1500.0, 3000.0]))
    np.testing.assert_allclose(x, [0.0, 1500.0, 3000.0])
    np.testing.assert_allclose(y, [0.0, 0.0, 0.0])
    # reach 1: polyline length 2 * hypot(1000, 250), hydraulic length 2000; s = 1000 is the middle vertex
    x, y = net.map_position(np.array([1, 1]), np.array([1000.0, 500.0]))
    np.testing.assert_allclose(x, [-1000.0, -1500.0])
    np.testing.assert_allclose(y, [-250.0, -375.0])


def test_map_position_inactive_and_missing_polylines():
    net = Network(ArrayHydraulicsProvider.from_dataset(three_reach_dataset()).static)
    x, y = net.map_position(np.array([-1, 0]), np.array([np.nan, 10.0]))
    assert np.isnan(x[0]) and np.isnan(y[0]) and np.isfinite(x[1])
    bare = Network(ArrayHydraulicsProvider.from_dataset(three_reach_dataset(polylines=None)).static)
    assert not bare.has_polylines
    x, y = bare.map_position(np.array([0]), np.array([10.0]))
    assert np.isnan(x[0]) and np.isnan(y[0])
    assert bare.polylines() == []


def test_polylines_list(net):
    pls = net.polylines()
    assert len(pls) == 3
    np.testing.assert_allclose(pls[2][0], [0.0, 1500.0, 3000.0])


def test_chain_upstream_of():
    net = Network(ArrayHydraulicsProvider.from_dataset(chain_dataset(n_reach=5)).static)
    assert list(net.headwaters()) == [1]
    assert sorted(net.upstream_of(3)) == [1, 2, 3]
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_network.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `network.py` (Network only; NetworkBins is Task 4)**

```python
"""Static river-network topology, polyline geometry, and sub-reach bins."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class Network:
    """Reach topology and geometry from a provider's static arrays.

    Args:
        static: mapping with at least reach_id, to_index, is_outlet, length; optionally the polyline block
            (vertex_x, vertex_y, vertex_dist, reach_vertex_start, reach_vertex_count).
        crs_wkt: CRS of the polylines, informational.
    """

    def __init__(self, static: Mapping[str, npt.NDArray[np.generic]], crs_wkt: str = "") -> None:
        self._static = static
        self.reach_id: IntArray = np.asarray(static["reach_id"], dtype=np.int64)
        self.to_index: npt.NDArray[np.int32] = np.asarray(static["to_index"], dtype=np.int32)
        self.length: FloatArray = np.asarray(static["length"], dtype=np.float64)
        self.is_outlet: npt.NDArray[np.bool_] = self.to_index < 0
        self.n_reach: int = int(self.reach_id.size)
        self.crs_wkt = crs_wkt
        self._index_by_id: dict[int, int] = {int(r): i for i, r in enumerate(self.reach_id)}
        self._parents_ptr: IntArray | None = None
        self._parents_idx: IntArray | None = None
        self._poly_index: tuple[FloatArray, IntArray, FloatArray, FloatArray] | None = None

    # ---- ids -------------------------------------------------------------
    def index_of(self, reach_id: int) -> int:
        """Index of a reach id; raises KeyError for an unknown id."""
        return self._index_by_id[int(reach_id)]

    def index_of_many(self, reach_ids: npt.ArrayLike) -> IntArray:
        """Indices of several reach ids; raises KeyError naming the first unknown id."""
        return np.array([self.index_of(r) for r in np.asarray(reach_ids).ravel()], dtype=np.int64)

    def id_of(self, index: int) -> int:
        """Reach id at an index."""
        return int(self.reach_id[index])

    # ---- topology ----------------------------------------------------------
    def _build_parents(self) -> tuple[IntArray, IntArray]:
        if self._parents_ptr is None:
            child = self.to_index.astype(np.int64)
            has = child >= 0
            order = np.argsort(child[has], kind="stable")
            src = np.nonzero(has)[0][order]
            counts = np.bincount(child[has], minlength=self.n_reach)
            self._parents_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            self._parents_idx = src.astype(np.int64)
        assert self._parents_idx is not None
        return self._parents_ptr, self._parents_idx

    def parents(self, index: int) -> IntArray:
        """Indices of reaches flowing directly into ``index``."""
        ptr, idx = self._build_parents()
        return idx[ptr[index] : ptr[index + 1]]

    def headwaters(self, as_index: bool = False) -> IntArray:
        """Reaches with no upstream reach (ids by default, indices with as_index)."""
        ptr, _ = self._build_parents()
        idx = np.nonzero(np.diff(ptr) == 0)[0].astype(np.int64)
        return idx if as_index else self.reach_id[idx]

    def outlets(self, as_index: bool = False) -> IntArray:
        """Reaches with to_index == -1 (ids by default, indices with as_index)."""
        idx = np.nonzero(self.is_outlet)[0].astype(np.int64)
        return idx if as_index else self.reach_id[idx]

    def upstream_of(self, reach_id: int) -> IntArray:
        """Ids of every reach draining to ``reach_id``, inclusive, in breadth-first order."""
        ptr, idx = self._build_parents()
        seen = [self.index_of(reach_id)]
        frontier = list(seen)
        while frontier:
            nxt: list[int] = []
            for i in frontier:
                nxt.extend(int(p) for p in idx[ptr[i] : ptr[i + 1]])
            seen.extend(nxt)
            frontier = nxt
        return self.reach_id[np.array(seen, dtype=np.int64)]

    # ---- geometry ----------------------------------------------------------
    @property
    def has_polylines(self) -> bool:
        """True when the static arrays carry the polyline block."""
        return "reach_vertex_start" in self._static

    def _build_poly_index(self) -> tuple[FloatArray, IntArray, FloatArray, FloatArray]:
        if self._poly_index is None:
            start = np.asarray(self._static["reach_vertex_start"], dtype=np.int64)
            count = np.asarray(self._static["reach_vertex_count"], dtype=np.int64)
            vdist = np.asarray(self._static["vertex_dist"], dtype=np.float64)
            reach_of_vertex = np.repeat(np.arange(self.n_reach), count)
            total = np.where(count > 0, vdist[np.clip(start + count - 1, 0, vdist.size - 1)], 0.0)
            stride = float(total.max()) + 1.0 if total.size else 1.0
            offset = np.arange(self.n_reach, dtype=np.float64) * stride
            # vertices of a reach are contiguous at start..start+count-1, in file order
            vertex_ids = np.concatenate([np.arange(s, s + c) for s, c in zip(start, count, strict=True)]).astype(np.int64)
            key = offset[reach_of_vertex] + vdist[vertex_ids]
            order = np.argsort(key, kind="stable")
            self._poly_index = (key[order], vertex_ids[order], total, offset)
        return self._poly_index

    def map_position(self, reach: npt.ArrayLike, s: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Map (reach, s) to x, y by scaling s / length onto the reach polyline's arc length.

        Args:
            reach: reach indices (-1 for inactive particles).
            s: distance from the upstream end (m); NaN for inactive particles.

        Returns:
            x and y arrays; NaN where there is no polyline, the reach has fewer than two vertices,
            or the particle is inactive.
        """
        r_all = np.asarray(reach, dtype=np.int64)
        s_all = np.asarray(s, dtype=np.float64)
        x = np.full(r_all.shape, np.nan)
        y = np.full(r_all.shape, np.nan)
        if not self.has_polylines or r_all.size == 0:
            return x, y
        sorted_key, sorted_vertex, total, offset = self._build_poly_index()
        start = np.asarray(self._static["reach_vertex_start"], dtype=np.int64)
        count = np.asarray(self._static["reach_vertex_count"], dtype=np.int64)
        vx = np.asarray(self._static["vertex_x"], dtype=np.float64)
        vy = np.asarray(self._static["vertex_y"], dtype=np.float64)
        vd = np.asarray(self._static["vertex_dist"], dtype=np.float64)
        ok = (r_all >= 0) & np.isfinite(s_all)
        ok[ok] &= count[r_all[ok]] >= 2
        if not ok.any():
            return x, y
        r = r_all[ok]
        frac = np.clip(s_all[ok] / self.length[r], 0.0, 1.0)
        target = frac * total[r]
        pos = np.searchsorted(sorted_key, offset[r] + target, side="right") - 1
        v0 = sorted_vertex[np.clip(pos, 0, sorted_vertex.size - 1)]
        last = start[r] + count[r] - 1
        v0 = np.clip(v0, start[r], last)
        v1 = np.minimum(v0 + 1, last)
        span = vd[v1] - vd[v0]
        w = np.where(span > 0.0, (target - vd[v0]) / np.where(span > 0.0, span, 1.0), 0.0)
        x[ok] = vx[v0] + w * (vx[v1] - vx[v0])
        y[ok] = vy[v0] + w * (vy[v1] - vy[v0])
        return x, y

    def polylines(self) -> list[tuple[FloatArray, FloatArray]]:
        """Per-reach (x, y) vertex arrays for plotting; empty when there are no polylines."""
        if not self.has_polylines:
            return []
        start = np.asarray(self._static["reach_vertex_start"], dtype=np.int64)
        count = np.asarray(self._static["reach_vertex_count"], dtype=np.int64)
        vx = np.asarray(self._static["vertex_x"], dtype=np.float64)
        vy = np.asarray(self._static["vertex_y"], dtype=np.float64)
        return [(vx[s : s + c], vy[s : s + c]) for s, c in zip(start, count, strict=True)]
```

- [ ] **Step 4: Run to verify pass, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_network.py -v` → 6 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/network.py tests/network/test_network.py
git commit -m "Add Network topology queries and polyline map positions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Sub-reach bins

**Files:**
- Modify: `src/fluvial_particle/network/network.py` (append `NetworkBins`)
- Test: `tests/network/test_network.py` (append)

**Interfaces:**
- Produces: `NetworkBins(network: Network, bin_length: float)` with `n_bins: int`, `bin_length: float`, `bins_per_reach: NDArray[int64]`, `reach_bin_start: NDArray[int64]`, `bin_reach: NDArray[int64]`, `bin_width: NDArray[float64]`, `s_start`, `s_end: NDArray[float64]`, `bin_of(reach, s) -> NDArray[int64]`, `midpoints_xy() -> tuple[NDArray, NDArray]`.

- [ ] **Step 1: Append failing tests to `tests/network/test_network.py`**

```python
from fluvial_particle.network.network import NetworkBins  # add to the imports at the top


def test_bins_layout(net):
    bins = NetworkBins(net, 400.0)
    assert list(bins.bins_per_reach) == [3, 5, 8]
    assert bins.n_bins == 16
    assert list(bins.reach_bin_start) == [0, 3, 8]
    np.testing.assert_allclose(bins.bin_width[:3], 1000.0 / 3)
    np.testing.assert_allclose(bins.bin_width[3:8], 400.0)
    np.testing.assert_allclose(bins.s_start[3:8], [0, 400, 800, 1200, 1600])
    np.testing.assert_allclose(bins.s_end[3:8], [400, 800, 1200, 1600, 2000])
    assert list(bins.bin_reach[8:10]) == [2, 2]


def test_bins_bin_of_edges(net):
    bins = NetworkBins(net, 400.0)
    assert list(bins.bin_of(np.array([1, 1, 2, 0]), np.array([799.9, 800.0, 3000.0, 0.0]))) == [4, 5, 15, 0]


def test_bins_one_per_reach(net):
    bins = NetworkBins(net, np.inf)
    assert bins.n_bins == 3
    assert list(bins.bin_of(np.array([0, 1, 2]), np.array([999.0, 1.0, 2999.0]))) == [0, 1, 2]
    x, y = bins.midpoints_xy()
    np.testing.assert_allclose(x[2], 1500.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_network.py -k bins -v`
Expected: FAIL with `ImportError: cannot import name 'NetworkBins'`

- [ ] **Step 3: Append `NetworkBins` to `network.py`**

```python
class NetworkBins:
    """Nearly uniform sub-reach bins for counts and concentration.

    Each reach is split into ``ceil(length / bin_length)`` bins of equal width within the reach.

    Args:
        network: the Network to discretize.
        bin_length: target bin length (m); ``np.inf`` gives one bin per reach.
    """

    def __init__(self, network: Network, bin_length: float) -> None:
        self.network = network
        self.bin_length = float(bin_length)
        length = network.length
        if np.isinf(self.bin_length):
            n = np.ones(network.n_reach, dtype=np.int64)
        else:
            if self.bin_length <= 0.0:
                raise ValueError("bin_length must be positive")
            n = np.maximum(1, np.ceil(length / self.bin_length)).astype(np.int64)
        self.bins_per_reach: IntArray = n
        self.reach_bin_start: IntArray = (np.cumsum(n) - n).astype(np.int64)
        self.n_bins: int = int(n.sum())
        self.bin_reach: IntArray = np.repeat(np.arange(network.n_reach, dtype=np.int64), n)
        self._width_per_reach: FloatArray = length / n
        self.bin_width: FloatArray = np.repeat(self._width_per_reach, n)
        local = np.arange(self.n_bins, dtype=np.float64) - self.reach_bin_start[self.bin_reach]
        self.s_start: FloatArray = local * self.bin_width
        self.s_end: FloatArray = (local + 1.0) * self.bin_width

    def bin_of(self, reach: npt.ArrayLike, s: npt.ArrayLike) -> IntArray:
        """Global bin index for (reach, s); s == length maps to the reach's last bin."""
        r = np.asarray(reach, dtype=np.int64)
        local = np.floor(np.asarray(s, dtype=np.float64) / self._width_per_reach[r]).astype(np.int64)
        local = np.clip(local, 0, self.bins_per_reach[r] - 1)
        return self.reach_bin_start[r] + local

    def midpoints_xy(self) -> tuple[FloatArray, FloatArray]:
        """Map coordinates of every bin midpoint (NaN without polylines)."""
        return self.network.map_position(self.bin_reach, 0.5 * (self.s_start + self.s_end))
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_network.py -v` → 9 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/network.py tests/network/test_network.py
git commit -m "Add NetworkBins sub-reach discretization

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: File provider: open, validate, static arrays, subsetting

**Files:**
- Create: `src/fluvial_particle/network/provider.py`
- Test: `tests/network/test_provider.py`

**Interfaces:**
- Consumes: `Network` (for outlet closure).
- Produces: `HydraulicsProvider` Protocol (`times`, `static`, `dtype`, `crs_wkt`, `n_reach`, `hydraulics(t)`, `close()`); `FileHydraulicsProvider(path, *, interpolation="linear", dtype="float64", reach_subset=None, time_window=None)` with the attributes above plus `path`, `interpolation`, `time_window` (settable property, tuple of datetime64 or None), `subset_index: NDArray[int64] | None`, `memory_estimate(n_particles) -> dict[str, int]`; module constants `REQUIRED_STATIC`, `REQUIRED_FIELDS`, `OPTIONAL_FIELDS`, `POLYLINE_VARS`.
- `hydraulics(t)` is implemented in Task 6; in this task it raises `NotImplementedError`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_provider.py`:

```python
"""Tests for FileHydraulicsProvider: validation, static arrays, subsetting."""

import numpy as np
import pytest
import xarray as xr

from fluvial_particle.network.provider import FileHydraulicsProvider
from tests.network.support import chain_dataset, three_reach_dataset, write_network_file


@pytest.fixture
def three_file(tmp_path):
    return write_network_file(tmp_path / "three.nc", three_reach_dataset(temperature=4.0))


def test_open_static_and_times(three_file):
    with FileHydraulicsProvider(three_file) as prov:
        assert prov.n_reach == 3
        assert prov.times.dtype == np.dtype("datetime64[ns]")
        assert prov.times.size == 3
        assert list(prov.static["reach_id"]) == [101, 102, 103]
        assert list(prov.static["to_index"]) == [2, 2, -1]
        assert prov.static["length"].dtype == np.float64
        assert prov.crs_wkt == 'PROJCRS["test"]'
        assert prov.dtype == np.dtype("float64")
        assert prov.subset_index is None
        assert prov.has_temperature


def test_polylines_are_lazy(three_file):
    with FileHydraulicsProvider(three_file) as prov:
        assert "reach_vertex_start" in prov.static
        assert not prov.static.polylines_loaded
        assert prov.static["vertex_x"].shape == (8,)
        assert prov.static.polylines_loaded
        assert list(prov.static["reach_vertex_start"]) == [0, 2, 5]


def test_missing_required_variable_lists_all(tmp_path):
    ds = three_reach_dataset().drop_vars(["velocity", "ustar"])
    path = write_network_file(tmp_path / "bad.nc", ds)
    with pytest.raises(ValueError, match="velocity") as exc:
        FileHydraulicsProvider(path)
    assert "ustar" in str(exc.value)


def test_wrong_units_raise_and_missing_units_warn(tmp_path):
    ds = three_reach_dataset()
    ds["velocity"].attrs["units"] = "ft s-1"
    with pytest.raises(ValueError, match="velocity.*units"):
        FileHydraulicsProvider(write_network_file(tmp_path / "u1.nc", ds))
    ds = three_reach_dataset()
    del ds["depth"].attrs["units"]
    with pytest.warns(UserWarning, match="depth"):
        FileHydraulicsProvider(write_network_file(tmp_path / "u2.nc", ds)).close()


def test_bad_to_index_and_cycle(tmp_path):
    ds = three_reach_dataset(to_index=[2, 5, -1])
    with pytest.raises(ValueError, match="to_index"):
        FileHydraulicsProvider(write_network_file(tmp_path / "range.nc", ds))
    ds = three_reach_dataset(to_index=[2, 2, 0])  # 0 -> 2 -> 0 cycle
    with pytest.raises(ValueError, match="cycle"):
        FileHydraulicsProvider(write_network_file(tmp_path / "cycle.nc", ds))


def test_is_outlet_inconsistent(tmp_path):
    ds = three_reach_dataset()
    ds["is_outlet"].values[:] = 0
    with pytest.raises(ValueError, match="is_outlet"):
        FileHydraulicsProvider(write_network_file(tmp_path / "outlet.nc", ds))


def test_polyline_block_validation(tmp_path):
    ds = three_reach_dataset()
    ds["reach_vertex_count"].values[2] = 10
    with pytest.raises(ValueError, match="reach_vertex"):
        FileHydraulicsProvider(write_network_file(tmp_path / "pl1.nc", ds)).static["vertex_x"]
    ds = three_reach_dataset()
    ds["vertex_dist"].values[6] = 5000.0  # not monotone inside reach 2
    with pytest.raises(ValueError, match="vertex_dist"):
        FileHydraulicsProvider(write_network_file(tmp_path / "pl2.nc", ds)).static["vertex_x"]


def test_time_not_increasing(tmp_path):
    t = np.array(["1979-01-02", "1979-01-01", "1979-01-03"], dtype="datetime64[ns]")
    ds = three_reach_dataset(times=t)
    with pytest.raises(ValueError, match="time"):
        FileHydraulicsProvider(write_network_file(tmp_path / "t.nc", ds))


def test_subset_by_ids_remaps_to_index(three_file):
    with FileHydraulicsProvider(three_file, reach_subset=[102, 101]) as prov:
        assert prov.n_reach == 2
        assert list(prov.static["reach_id"]) == [101, 102]  # file order kept
        assert list(prov.static["to_index"]) == [-1, -1]  # downstream reach dropped -> outlet
        assert list(prov.static["is_outlet"]) == [1, 1]
        assert list(prov.subset_index) == [0, 1]
        assert prov.static["vertex_x"].shape == (5,)
        assert list(prov.static["reach_vertex_start"]) == [0, 2]
    with pytest.raises(KeyError):
        FileHydraulicsProvider(three_file, reach_subset=[999])


def test_subset_by_outlet_closure(tmp_path):
    path = write_network_file(tmp_path / "chain.nc", chain_dataset(n_reach=6))
    with FileHydraulicsProvider(path, reach_subset={"outlet": 4}) as prov:
        assert list(prov.static["reach_id"]) == [1, 2, 3, 4]
        assert list(prov.static["to_index"]) == [1, 2, 3, -1]


def test_chunk_warning(tmp_path):
    ds = three_reach_dataset()
    path = tmp_path / "chunked.nc"
    ds.to_netcdf(path, engine="h5netcdf", encoding={"velocity": {"chunksizes": (3, 1)}})
    with pytest.warns(UserWarning, match="chunk"):
        FileHydraulicsProvider(path).close()


def test_memory_estimate(three_file):
    with FileHydraulicsProvider(three_file, dtype="float32") as prov:
        est = prov.memory_estimate(1000)
        assert est["window_bytes"] == 2 * 3 * 7 * 4  # 2 slices, 3 reaches, 6 fields + temperature, float32
        assert est["particle_bytes"] > 0
        assert est["static_bytes"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `provider.py` (validation, static, subsetting; `hydraulics` raises NotImplementedError)**

```python
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


REQUIRED_STATIC: dict[str, str | None] = {"reach_id": None, "to_index": None, "is_outlet": None, "length": "m", "slope": "m m-1"}
OPTIONAL_STATIC: tuple[str, ...] = ("mann_n", "elevation_mid", "bankfull_width", "bankfull_depth", "x_mid", "y_mid")
REQUIRED_FIELDS: dict[str, str] = {
    "flow_in": "m3 s-1", "flow_out": "m3 s-1", "velocity": "m s-1", "depth": "m", "width": "m", "ustar": "m s-1",
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
        self._data = data
        self._lazy_keys = tuple(lazy_keys)
        self._loader = loader
        self.polylines_loaded = not self._lazy_keys

    def __getitem__(self, key: str) -> npt.NDArray[Any]:
        if key not in self._data and key in self._lazy_keys and not self.polylines_loaded:
            self._data.update(self._loader())
            self.polylines_loaded = True
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter({**dict.fromkeys(self._data), **dict.fromkeys(self._lazy_keys)})

    def __len__(self) -> int:
        return len(set(self._data) | set(self._lazy_keys))

    def __contains__(self, key: object) -> bool:
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
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying dataset."""
        self._ds.close()

    # ---- validation --------------------------------------------------------
    def _validate_schema(self) -> None:
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
        out: dict[str, npt.NDArray[Any]] = {}
        for name in list(REQUIRED_STATIC) + [v for v in OPTIONAL_STATIC if v in self._ds]:
            arr = self._ds[name].values
            if name in ("reach_id",):
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

    def _validate_polylines(self, start: npt.NDArray[np.int64], count: npt.NDArray[np.int64], vdist: FloatArray) -> None:
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
            "vertex_x": vx, "vertex_y": vy, "vertex_dist": vd,
            "reach_vertex_start": start, "reach_vertex_count": count.astype(np.int32),
        }

    # ---- housekeeping ------------------------------------------------------
    def _check_chunking(self) -> None:
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
        t0, t1 = (np.datetime64(v, "ns") for v in value)
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
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_provider.py -v` → 13 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/provider.py tests/network/test_provider.py
git commit -m "Add FileHydraulicsProvider validation, static arrays, and reach subsetting

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: File provider: streaming window and time interpolation

**Files:**
- Modify: `src/fluvial_particle/network/provider.py` (replace `hydraulics`, add `_bracket`, `_read_slice`, `_ensure_window`)
- Test: `tests/network/test_provider.py` (append)

**Interfaces:**
- Produces: `FileHydraulicsProvider.hydraulics(t) -> dict[str, FloatArray]` with keys `flow_in`, `flow_out`, `velocity`, `depth`, `width`, `ustar` (+ `water_temperature`), each `(n_reach,)` in `self.dtype`; `window_indices -> tuple[int, int | None]` for tests.

- [ ] **Step 1: Append failing tests**

```python
def _stepped_file(tmp_path, n_time=4):
    n = 3
    vel = np.arange(1, n_time + 1, dtype=float)[:, None] * np.ones((1, n))  # day k has velocity k+1
    flow = np.full((n_time, n), 10.0)
    t = np.datetime64("1979-01-01", "ns") + np.arange(n_time) * np.timedelta64(1, "D")
    return write_network_file(tmp_path / "step.nc", three_reach_dataset(velocity=vel, flow_out=flow, times=t))


def test_hold_and_linear(tmp_path):
    path = _stepped_file(tmp_path)
    noon = np.datetime64("1979-01-02T12:00", "ns")
    with FileHydraulicsProvider(path, interpolation="hold") as prov:
        assert prov.hydraulics(noon)["velocity"][0] == 2.0
        assert prov.hydraulics(np.datetime64("1979-01-04", "ns"))["velocity"][0] == 4.0
    with FileHydraulicsProvider(path, interpolation="linear") as prov:
        assert prov.hydraulics(noon)["velocity"][0] == pytest.approx(2.5)
        assert prov.hydraulics(np.datetime64("1979-01-04", "ns"))["velocity"][0] == 4.0
        h = prov.hydraulics(np.datetime64("1979-01-04", "ns"))
        assert set(h) == {"flow_in", "flow_out", "velocity", "depth", "width", "ustar"}
        assert h["velocity"].dtype == np.float64


def test_linear_zero_flow_guard(tmp_path):
    n = 3
    flow = np.array([[10.0, 0.0, 10.0], [10.0, 10.0, 0.0]])
    vel = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    t = np.datetime64("1979-01-01", "ns") + np.arange(2) * np.timedelta64(1, "D")
    path = write_network_file(tmp_path / "dry.nc", three_reach_dataset(velocity=vel, flow_out=flow, times=t))
    with FileHydraulicsProvider(path) as prov:
        h = prov.hydraulics(np.datetime64("1979-01-01T12:00", "ns"))
        assert h["velocity"][0] == 1.0
        assert h["velocity"][1] == 0.0 and h["ustar"][1] == 0.0
        assert h["velocity"][2] == 0.0 and h["ustar"][2] == 0.0
        assert h["flow_out"][1] == pytest.approx(5.0)  # flow itself still interpolates


def test_window_advances_one_read_at_a_time(tmp_path, monkeypatch):
    path = _stepped_file(tmp_path, n_time=5)
    with FileHydraulicsProvider(path) as prov:
        reads = []
        orig = prov._read_slice
        monkeypatch.setattr(prov, "_read_slice", lambda k: (reads.append(k), orig(k))[1])
        day = np.timedelta64(1, "D")
        t0 = np.datetime64("1979-01-01", "ns")
        prov.hydraulics(t0 + np.timedelta64(6, "h"))
        assert reads == [0, 1]
        assert prov.window_indices == (0, 1)
        prov.hydraulics(t0 + np.timedelta64(18, "h"))
        assert reads == [0, 1]
        prov.hydraulics(t0 + day + np.timedelta64(1, "h"))
        assert reads == [0, 1, 2]
        assert prov.window_indices == (1, 2)
        prov.hydraulics(t0 + 3 * day + np.timedelta64(1, "h"))  # skip ahead: two reads
        assert reads == [0, 1, 2, 3, 4]
        prov.hydraulics(t0 + np.timedelta64(1, "h"))  # backwards: reset, two reads
        assert reads == [0, 1, 2, 3, 4, 0, 1]


def test_out_of_range_and_time_window(tmp_path):
    path = _stepped_file(tmp_path)
    with FileHydraulicsProvider(path) as prov:
        with pytest.raises(ValueError, match="outside"):
            prov.hydraulics(np.datetime64("1978-12-31", "ns"))
        prov.time_window = (np.datetime64("1979-01-02", "ns"), np.datetime64("1979-01-03", "ns"))
        with pytest.raises(ValueError, match="outside"):
            prov.hydraulics(np.datetime64("1979-01-03T01:00", "ns"))
        with pytest.raises(ValueError, match="time_window"):
            prov.time_window = (np.datetime64("1979-01-03", "ns"), np.datetime64("1979-01-02", "ns"))


def test_subset_and_dtype_apply_to_fields(tmp_path):
    path = _stepped_file(tmp_path)
    with FileHydraulicsProvider(path, reach_subset=[103], dtype="float32") as prov:
        h = prov.hydraulics(np.datetime64("1979-01-01", "ns"))
        assert h["velocity"].shape == (1,)
        assert h["velocity"].dtype == np.float32


def test_single_timestamp_file(tmp_path):
    t = np.array(["1979-01-01"], dtype="datetime64[ns]")
    path = write_network_file(tmp_path / "one.nc", three_reach_dataset(times=t))
    with FileHydraulicsProvider(path) as prov:
        assert prov.hydraulics(t[0])["velocity"][0] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_provider.py -k "hold or guard or window or range or dtype or single" -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Replace the `hydraulics` stub in `provider.py`**

```python
    # ---- streaming ---------------------------------------------------------
    @property
    def window_indices(self) -> tuple[int, int | None]:
        """Indices of the two loaded time slices (second is None at the end of the axis)."""
        k1 = self._window_k + 1 if self._window_k + 1 < self._n_time else None
        return self._window_k, k1

    def _bracket(self, t: np.datetime64) -> int:
        k = int(np.searchsorted(self.times, t, side="right")) - 1
        if self.interpolation == "linear" and k == self._n_time - 1 and self._n_time > 1:
            k -= 1
        return k

    def _read_slice(self, k: int) -> dict[str, FloatArray]:
        out: dict[str, FloatArray] = {}
        for name in self._field_names:
            arr = self._ds[name][k].values
            if self.subset_index is not None:
                arr = arr[self.subset_index]
            out[name] = np.ascontiguousarray(arr, dtype=self.dtype)
        return out

    def _ensure_window(self, k: int) -> None:
        if k == self._window_k:
            return
        k1 = k + 1 if k + 1 < self._n_time else None
        new: dict[int, dict[str, FloatArray]] = {}
        new[k] = self._slices[k] if k in self._slices else self._read_slice(k)
        if k1 is not None:
            new[k1] = self._slices[k1] if k1 in self._slices else self._read_slice(k1)
        self._slices = new
        self._window_k = k

    def hydraulics(self, t: np.datetime64) -> dict[str, FloatArray]:
        """Per-reach hydraulics at ``t`` (hold or linear); velocity and ustar are 0 across dry intervals.

        Args:
            t: requested time; must be within the file range and the time_window when set.

        Returns:
            dict of (n_reach,) arrays keyed by the export's variable names.

        Raises:
            ValueError: ``t`` outside the allowed range.
        """
        tt = np.datetime64(t, "ns")
        lo, hi = self._time_window if self._time_window is not None else (self.times[0], self.times[-1])
        if tt < lo or tt > hi:
            raise ValueError(f"time {tt} is outside the provider range {lo}..{hi}")
        k = self._bracket(tt)
        self._ensure_window(k)
        a = self._slices[k]
        k1 = k + 1 if k + 1 < self._n_time else None
        if self.interpolation == "hold" or k1 is None:
            return {name: arr.copy() for name, arr in a.items()}
        b = self._slices[k1]
        span = (self.times[k1] - self.times[k]) / np.timedelta64(1, "s")
        f = float((tt - self.times[k]) / np.timedelta64(1, "s")) / float(span)
        out = {name: (a[name] + f * (b[name] - a[name])).astype(self.dtype, copy=False) for name in a}
        dry = (a["flow_out"] <= 0.0) | (b["flow_out"] <= 0.0)
        out["velocity"][dry] = 0.0
        out["ustar"][dry] = 0.0
        return out
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_provider.py -v` → 19 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/provider.py tests/network/test_provider.py
git commit -m "Stream hydraulics through a two-slice window with hold or linear interpolation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 7: Configuration

**Files:**
- Create: `src/fluvial_particle/network/config.py`
- Test: `tests/network/test_config.py`

**Interfaces:**
- Consumes: `DISPERSION_MODELS` from `dispersion.py`, `INTERPOLATIONS` from `provider.py`.
- Produces: `parse_datetime(value) -> np.datetime64` (ns); `DispersionConfig(model="fischer", scale=1.0, cap=None, value=None)` with `from_dict`, `to_dict`; `NetworkConfig(...)` frozen dataclass with fields `hydraulics_file: str`, `sources: tuple[dict[str, Any], ...]`, `interpolation`, `reach_subset`, `dtype`, `start_time`, `end_time`, `dt`, `output_interval`, `dispersion`, `particle_mass`, `mass_units`, `max_hops`, `seed`; classmethods `from_dict(d)`, `from_toml(path)`, `coerce(obj)`; methods `resolve_times(times) -> tuple[np.datetime64, np.datetime64]`, `to_dict() -> dict`; `get_network_config_template() -> str`; constant `SOURCE_FORMS = ("slug", "loading", "concentration")`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_config.py`:

```python
"""Tests for NetworkConfig."""

import json
import sys

import numpy as np
import pytest

from fluvial_particle.network.config import DispersionConfig, NetworkConfig, get_network_config_template, parse_datetime

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

MINIMAL = {"hydraulics_file": "net.nc", "particle_mass": 1.0, "sources": [{"reach_id": 101, "form": "slug", "time": 0.0, "mass": 5.0}]}


def test_defaults_and_parse():
    cfg = NetworkConfig.from_dict(MINIMAL)
    assert cfg.dt == 900.0 and cfg.output_interval == 3600.0
    assert cfg.interpolation == "linear" and cfg.dtype == "float64"
    assert cfg.dispersion == DispersionConfig()
    assert cfg.sources[0]["reach_id"] == 101
    assert cfg.start_time is None


def test_parse_datetime_forms():
    assert parse_datetime("1979-03-01") == np.datetime64("1979-03-01T00:00:00", "ns")
    assert parse_datetime(np.datetime64("1979-03-01T06:00")) == np.datetime64("1979-03-01T06:00", "ns")
    import datetime as dtm
    assert parse_datetime(dtm.datetime(1979, 3, 1, 6)) == np.datetime64("1979-03-01T06:00", "ns")


def test_unknown_key_and_bad_values():
    with pytest.raises(ValueError, match="unknown"):
        NetworkConfig.from_dict({**MINIMAL, "bogus": 1})
    with pytest.raises(ValueError, match="output_interval"):
        NetworkConfig.from_dict({**MINIMAL, "dt": 900.0, "output_interval": 1000.0})
    with pytest.raises(ValueError, match="interpolation"):
        NetworkConfig.from_dict({**MINIMAL, "interpolation": "cubic"})
    with pytest.raises(ValueError, match="sources"):
        NetworkConfig.from_dict({**MINIMAL, "sources": []})
    with pytest.raises(ValueError, match="form"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{"reach_id": 1, "form": "leak"}]})
    with pytest.raises(ValueError, match="particle"):
        NetworkConfig.from_dict({"hydraulics_file": "n.nc", "sources": [{"reach_id": 1, "form": "slug", "time": 0, "mass": 1}]})
    with pytest.raises(ValueError, match="s_frac"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{**MINIMAL["sources"][0], "s": 1.0, "s_frac": 0.5}]})
    with pytest.raises(ValueError, match="dispersion"):
        NetworkConfig.from_dict({**MINIMAL, "dispersion": {"model": "constant"}})
    with pytest.raises(ValueError, match="reach_subset"):
        NetworkConfig.from_dict({**MINIMAL, "reach_subset": {"inlet": 3}})


def test_resolve_times():
    times = np.datetime64("1979-01-01", "ns") + np.arange(5) * np.timedelta64(1, "D")
    cfg = NetworkConfig.from_dict(MINIMAL)
    assert cfg.resolve_times(times) == (times[0], times[-1])
    cfg = NetworkConfig.from_dict({**MINIMAL, "start_time": "1979-01-02", "end_time": "1979-01-03T12:00"})
    assert cfg.resolve_times(times) == (np.datetime64("1979-01-02", "ns"), np.datetime64("1979-01-03T12:00", "ns"))
    with pytest.raises(ValueError, match="end_time"):
        NetworkConfig.from_dict({**MINIMAL, "end_time": "1979-02-01"}).resolve_times(times)
    with pytest.raises(ValueError, match="start_time"):
        NetworkConfig.from_dict({**MINIMAL, "start_time": "1979-01-03", "end_time": "1979-01-02"})


def test_toml_round_trip(tmp_path):
    text = get_network_config_template()
    parsed = tomllib.loads(text)
    assert "network" in parsed and parsed["network"]["sources"]
    path = tmp_path / "s.toml"
    path.write_text(text)
    cfg = NetworkConfig.from_toml(path)
    assert cfg.hydraulics_file.endswith(".nc")
    d = cfg.to_dict()
    json.dumps(d)  # JSON-safe
    assert NetworkConfig.from_dict(d) == cfg
    assert NetworkConfig.coerce(path) == cfg
    assert NetworkConfig.coerce(cfg) is cfg
    (tmp_path / "empty.toml").write_text("x = 1\n")
    with pytest.raises(ValueError, match=r"\[network\]"):
        NetworkConfig.from_toml(tmp_path / "empty.toml")
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `config.py`**

```python
"""Configuration for the 1D network particle solver."""

from __future__ import annotations

import dataclasses
import datetime as dtm
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .dispersion import DISPERSION_MODELS
from .provider import INTERPOLATIONS


if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SOURCE_FORMS = ("slug", "loading", "concentration")
DTYPES = ("float32", "float64")


def parse_datetime(value: str | dtm.datetime | np.datetime64) -> np.datetime64:
    """Parse an ISO string, datetime, or datetime64 into datetime64[ns]."""
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    if isinstance(value, dtm.datetime):
        return np.datetime64(value.replace(tzinfo=None), "ns")
    return np.datetime64(str(value), "ns")


@dataclasses.dataclass(frozen=True)
class DispersionConfig:
    """Longitudinal dispersion settings.

    Args:
        model: "fischer", "constant", or "none".
        scale: multiplier on the Fischer coefficient.
        cap: upper bound on K (m^2/s).
        value: K (m^2/s) for the constant model.
    """

    model: str = "fischer"
    scale: float = 1.0
    cap: float | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        if self.model not in DISPERSION_MODELS:
            raise ValueError(f"dispersion model must be one of {DISPERSION_MODELS}, got {self.model!r}")
        if self.scale <= 0.0:
            raise ValueError("dispersion scale must be positive")
        if self.cap is not None and self.cap <= 0.0:
            raise ValueError("dispersion cap must be positive")
        if self.model == "constant" and (self.value is None or self.value < 0.0):
            raise ValueError("dispersion model 'constant' requires a non-negative value")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DispersionConfig:
        """Build from a mapping; unknown keys raise."""
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown dispersion keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict."""
        return dataclasses.asdict(self)


def _validate_source(i: int, row: Mapping[str, Any], particle_mass: float | None) -> dict[str, Any]:
    if "reach_id" not in row:
        raise ValueError(f"sources[{i}] needs a reach_id")
    form = row.get("form")
    if form not in SOURCE_FORMS:
        raise ValueError(f"sources[{i}] form must be one of {SOURCE_FORMS}, got {form!r}")
    if "s" in row and "s_frac" in row:
        raise ValueError(f"sources[{i}] may set s or s_frac, not both")
    if "particles" in row and int(row["particles"]) < 1:
        raise ValueError(f"sources[{i}] particles must be a positive integer")
    if "particles" not in row and particle_mass is None:
        raise ValueError(f"sources[{i}] needs particles, or set a global particle_mass")
    return dict(row)


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    """Settings for one network particle run (see the [network] TOML table)."""

    hydraulics_file: str
    sources: tuple[dict[str, Any], ...]
    interpolation: str = "linear"
    reach_subset: tuple[int, ...] | dict[str, int] | None = None
    dtype: str = "float64"
    start_time: np.datetime64 | None = None
    end_time: np.datetime64 | None = None
    dt: float = 900.0
    output_interval: float = 3600.0
    dispersion: DispersionConfig = dataclasses.field(default_factory=DispersionConfig)
    particle_mass: float | None = None
    mass_units: str = "kg"
    max_hops: int = 1000
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {self.interpolation!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        ratio = self.output_interval / self.dt
        if self.output_interval <= 0.0 or abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("output_interval must be a positive integer multiple of dt")
        if self.particle_mass is not None and self.particle_mass <= 0.0:
            raise ValueError("particle_mass must be positive")
        if self.max_hops < 1:
            raise ValueError("max_hops must be at least 1")
        if not self.sources:
            raise ValueError("sources must contain at least one source")
        object.__setattr__(
            self, "sources", tuple(_validate_source(i, r, self.particle_mass) for i, r in enumerate(self.sources))
        )
        if isinstance(self.reach_subset, Mapping):
            if set(self.reach_subset) != {"outlet"}:
                raise ValueError("reach_subset mapping must be {'outlet': reach_id}")
            object.__setattr__(self, "reach_subset", {"outlet": int(self.reach_subset["outlet"])})
        elif self.reach_subset is not None:
            object.__setattr__(self, "reach_subset", tuple(int(r) for r in self.reach_subset))
        for name in ("start_time", "end_time"):
            v = getattr(self, name)
            if v is not None:
                object.__setattr__(self, name, parse_datetime(v))
        if self.start_time is not None and self.end_time is not None and self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> NetworkConfig:
        """Build from a plain mapping such as the [network] table; unknown keys raise."""
        data = dict(d)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown network config keys: {sorted(unknown)}")
        disp = data.get("dispersion", {})
        data["dispersion"] = disp if isinstance(disp, DispersionConfig) else DispersionConfig.from_dict(disp)
        data["sources"] = tuple(dict(r) for r in data.get("sources", ()))
        return cls(**data)

    @classmethod
    def from_toml(cls, path: str | pathlib.Path) -> NetworkConfig:
        """Read the [network] table of a TOML file."""
        with pathlib.Path(path).open("rb") as f:
            doc = tomllib.load(f)
        if "network" not in doc:
            raise ValueError(f"{path} has no [network] table")
        return cls.from_dict(doc["network"])

    @classmethod
    def coerce(cls, obj: NetworkConfig | Mapping[str, Any] | str | pathlib.Path) -> NetworkConfig:
        """Accept a NetworkConfig, a mapping, or a TOML path."""
        if isinstance(obj, NetworkConfig):
            return obj
        if isinstance(obj, Mapping):
            return cls.from_dict(obj)
        return cls.from_toml(obj)

    def resolve_times(self, times: npt.NDArray[np.datetime64]) -> tuple[np.datetime64, np.datetime64]:
        """Start and end of the run, defaulting to the provider's first and last timestamps.

        Raises:
            ValueError: the run window lies outside the provider's time axis.
        """
        start = times[0] if self.start_time is None else self.start_time
        end = times[-1] if self.end_time is None else self.end_time
        if start < times[0] or start > times[-1]:
            raise ValueError(f"start_time {start} is outside the hydraulics time axis {times[0]}..{times[-1]}")
        if end < times[0] or end > times[-1]:
            raise ValueError(f"end_time {end} is outside the hydraulics time axis {times[0]}..{times[-1]}")
        if start >= end:
            raise ValueError("start_time must be before end_time")
        return start, end

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (datetimes as ISO strings) that from_dict accepts."""
        d = dataclasses.asdict(self)
        d["sources"] = [dict(r) for r in self.sources]
        d["dispersion"] = self.dispersion.to_dict()
        for name in ("start_time", "end_time"):
            if d[name] is not None:
                d[name] = str(np.datetime64(d[name], "s"))
        if isinstance(self.reach_subset, tuple):
            d["reach_subset"] = list(self.reach_subset)
        return d


NETWORK_TEMPLATE = '''# fluvial-particle network solver settings
# All keys live in the [network] table. Times are ISO datetimes or seconds from start_time.

[network]
hydraulics_file = "drb_network_hydraulics.nc"   # pywatershed network hydraulics export
interpolation = "linear"      # "linear" between daily values, or "hold"
dtype = "float64"             # "float32" for very large networks
# reach_subset = [4205, 4204]  # reach ids to keep; or {outlet = 4205} for everything upstream of one reach
# start_time = "1979-03-01"    # default: first timestamp in the file
# end_time = "1979-04-01"      # default: last timestamp in the file
dt = 900.0                    # solver step (s)
output_interval = 3600.0      # output every n*dt seconds
particle_mass = 1.0           # mass per particle; sources may override with particles = N
mass_units = "kg"             # label only; concentrations are reported in mass_units m-3
max_hops = 1000
# seed = 42

[network.dispersion]
model = "fischer"             # "fischer", "constant", or "none"
scale = 1.0                   # multiplier on the Fischer coefficient
# cap = 1000.0                # optional upper bound (m2/s)
# value = 10.0                # K for model = "constant"

# Sources: a slug (instantaneous mass), a loading (mass rate), or a concentration at the release point.
[[network.sources]]
reach_id = 1234
form = "slug"
time = 0.0
mass = 1000.0

[[network.sources]]
reach_id = 2345
s_frac = 0.5                  # release mid-reach (or s = meters from the upstream end)
form = "loading"
rate = 0.01                   # mass_units per second
start = 0.0
end = 86400.0
# spacing = "even"            # or "poisson"

[[network.sources]]
reach_id = 3456
form = "concentration"        # mass_units per m3 at the release point, times the reach flow
curve = [[0.0, 0.0], [3600.0, 5.0], [7200.0, 0.0]]
particles = 500               # fixed count for this source
'''


def get_network_config_template() -> str:
    """The commented TOML template for a network run."""
    return NETWORK_TEMPLATE
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_config.py -v` → 5 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/config.py tests/network/test_config.py
git commit -m "Add NetworkConfig with TOML loading and validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Sources and the particle schedule

**Files:**
- Create: `src/fluvial_particle/network/sources.py`
- Test: `tests/network/test_sources.py`

**Interfaces:**
- Consumes: `Network`, `HydraulicsProvider`, `parse_datetime`.
- Produces: `ParticleSchedule` dataclass (`release_reach: NDArray[int32]`, `release_s: NDArray[float64]`, `release_time: NDArray[float64]`, `mass: NDArray[float64]`, `source_index: NDArray[int32]`, property `n`, method `slice(lo, hi)`, classmethod `concat(parts)`, classmethod `simple(reach, s, time, mass=1.0, source_index=0)`); `seconds_from_start(value, start_time) -> float`; `expand_sources(sources, network, provider, *, start_time, end_time, particle_mass, rng) -> ParticleSchedule`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_sources.py`:

```python
"""Tests for source expansion into a particle schedule."""

import numpy as np
import pytest

from fluvial_particle.network.network import Network
from fluvial_particle.network.sources import ParticleSchedule, expand_sources, seconds_from_start
from tests.network.support import ArrayHydraulicsProvider, three_reach_dataset

T0 = np.datetime64("1979-01-01", "ns")
T1 = np.datetime64("1979-01-03", "ns")
DAY = 86400.0


@pytest.fixture
def env():
    ds = three_reach_dataset(flow_in=[10.0, 5.0, 60.0], flow_out=[10.0, 5.0, 80.0])
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    return Network(prov.static), prov


def expand(env, sources, particle_mass=None, seed=0):
    net, prov = env
    return expand_sources(sources, net, prov, start_time=T0, end_time=T1, particle_mass=particle_mass, rng=np.random.RandomState(seed))


def test_seconds_from_start():
    assert seconds_from_start(30.0, T0) == 30.0
    assert seconds_from_start("1979-01-02", T0) == DAY


def test_slug_global_particle_mass(env):
    sch = expand(env, [{"reach_id": 101, "form": "slug", "time": 10.0, "mass": 2.5}], particle_mass=1.0)
    assert sch.n == 2  # round(2.5) = 2, last absorbs remainder
    np.testing.assert_allclose(sch.mass, [1.0, 1.5])
    assert list(sch.release_reach) == [0, 0]
    assert list(sch.release_time) == [10.0, 10.0]
    assert list(sch.release_s) == [0.0, 0.0]
    assert list(sch.source_index) == [0, 0]


def test_slug_fixed_particles_and_position(env):
    sch = expand(env, [{"reach_id": 102, "form": "slug", "time": "1979-01-02", "mass": 3.0, "particles": 4, "s_frac": 0.25}])
    assert sch.n == 4
    np.testing.assert_allclose(sch.mass, 0.75)
    np.testing.assert_allclose(sch.release_s, 500.0)
    assert list(sch.release_time) == [DAY] * 4
    sch = expand(env, [{"reach_id": 102, "form": "slug", "time": 0.0, "mass": 3.0, "particles": 1, "s": 1999.0}])
    assert sch.release_s[0] == 1999.0


def test_constant_loading_even_spacing(env):
    row = {"reach_id": 101, "form": "loading", "rate": 0.01, "start": 0.0, "end": 1000.0, "particles": 4}
    sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 10.0)
    np.testing.assert_allclose(sch.release_time, [125.0, 375.0, 625.0, 875.0])


def test_ramp_curve_quantiles(env):
    # rate rises linearly 0 -> 1 over 1000 s: cumulative M(t) = t^2 / 2000, total 500
    row = {"reach_id": 101, "form": "loading", "curve": [[0.0, 0.0], [1000.0, 1.0]], "particles": 2}
    sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 500.0)
    # quantiles at M = 125 and 375 -> t = sqrt(2000 * M)
    np.testing.assert_allclose(sch.release_time, np.sqrt(2000.0 * np.array([125.0, 375.0])))


def test_concentration_uses_interpolated_flow(env):
    # reach 103: flow_in 60, flow_out 80, s_frac 0.5 -> Q = 70; C = 2 for 100 s -> M = 14000
    row = {"reach_id": 103, "form": "concentration", "value": 2.0, "start": 0.0, "end": 100.0, "s_frac": 0.5, "particles": 7}
    sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 14000.0)
    row = {"reach_id": 103, "form": "concentration", "curve": [[0.0, 2.0], [100.0, 2.0]], "particles": 7}
    sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 100.0 * 2.0 * 60.0)  # s = 0 -> flow_in


def test_poisson_spacing_reproducible(env):
    row = {"reach_id": 101, "form": "loading", "rate": 0.01, "start": 0.0, "end": 1000.0, "spacing": "poisson"}
    a = expand(env, [row], particle_mass=0.5, seed=3)
    b = expand(env, [row], particle_mass=0.5, seed=3)
    assert a.n == b.n and a.n > 0
    np.testing.assert_allclose(a.release_time, b.release_time)
    assert np.all(np.diff(a.release_time) >= 0)
    np.testing.assert_allclose(a.mass.sum(), 10.0)


def test_multiple_sources_concatenate(env):
    sch = expand(env, [
        {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 2},
        {"reach_id": 102, "form": "slug", "time": 5.0, "mass": 1.0, "particles": 3},
    ])
    assert sch.n == 5
    assert list(sch.source_index) == [0, 0, 1, 1, 1]
    part = sch.slice(1, 4)
    assert part.n == 3 and list(part.source_index) == [0, 1, 1]


def test_errors_and_truncation(env):
    with pytest.raises(KeyError):
        expand(env, [{"reach_id": 999, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 1}])
    with pytest.raises(ValueError, match="s"):
        expand(env, [{"reach_id": 101, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 1, "s": 5000.0}])
    with pytest.raises(ValueError, match="window"):
        expand(env, [{"reach_id": 101, "form": "slug", "time": 3 * DAY, "mass": 1.0, "particles": 1}])
    with pytest.raises(ValueError, match="mass"):
        expand(env, [{"reach_id": 101, "form": "loading", "rate": 0.0, "start": 0.0, "end": 10.0, "particles": 1}])
    with pytest.warns(UserWarning, match="window"):
        sch = expand(env, [{"reach_id": 101, "form": "loading", "rate": 1.0, "start": DAY, "end": 3 * DAY, "particles": 2}])
    np.testing.assert_allclose(sch.mass.sum(), DAY)  # only the day inside the window


def test_schedule_simple_and_concat():
    a = ParticleSchedule.simple(reach=0, s=10.0, time=0.0)
    b = ParticleSchedule.simple(reach=1, s=0.0, time=5.0, mass=2.0, source_index=1)
    c = ParticleSchedule.concat([a, b])
    assert c.n == 2 and list(c.mass) == [1.0, 2.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `sources.py`**

```python
"""Mass-loading sources and their expansion into a per-particle release schedule."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .config import parse_datetime
from .network import Network
from .provider import HydraulicsProvider


FloatArray = npt.NDArray[np.float64]


@dataclasses.dataclass
class ParticleSchedule:
    """Per-particle release reach, position, time (seconds from start), mass, and source row."""

    release_reach: npt.NDArray[np.int32]
    release_s: FloatArray
    release_time: FloatArray
    mass: FloatArray
    source_index: npt.NDArray[np.int32]

    @property
    def n(self) -> int:
        """Number of particles."""
        return int(self.release_reach.size)

    def slice(self, lo: int, hi: int) -> ParticleSchedule:
        """Particles lo..hi-1 (an MPI rank's slice)."""
        return ParticleSchedule(*(getattr(self, f.name)[lo:hi] for f in dataclasses.fields(self)))

    @classmethod
    def concat(cls, parts: Sequence[ParticleSchedule]) -> ParticleSchedule:
        """Concatenate schedules in order."""
        return cls(*(np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(cls)))

    @classmethod
    def simple(cls, reach: int, s: float, time: float, mass: float = 1.0, source_index: int = 0) -> ParticleSchedule:
        """A one-particle schedule, handy in tests."""
        return cls(
            np.array([reach], dtype=np.int32), np.array([s], dtype=np.float64), np.array([time], dtype=np.float64),
            np.array([mass], dtype=np.float64), np.array([source_index], dtype=np.int32),
        )


def seconds_from_start(value: Any, start_time: np.datetime64) -> float:
    """Seconds from ``start_time``: numbers pass through, strings and datetimes are parsed."""
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    return float((parse_datetime(value) - start_time) / np.timedelta64(1, "s"))


def _resolve_position(i: int, row: Mapping[str, Any], network: Network) -> tuple[int, float]:
    idx = network.index_of(int(row["reach_id"]))
    length = float(network.length[idx])
    if "s_frac" in row:
        frac = float(row["s_frac"])
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"sources[{i}] s_frac must be in [0, 1]")
        return idx, frac * length
    s = float(row.get("s", 0.0))
    if not 0.0 <= s <= length:
        raise ValueError(f"sources[{i}] s = {s} is outside [0, {length}] for reach {row['reach_id']}")
    return idx, s


def _curve(i: int, row: Mapping[str, Any], key: str, start_time: np.datetime64, total: float) -> tuple[FloatArray, FloatArray]:
    """Breakpoints (t, value) of a constant-with-window or tabulated curve, restricted to [0, total]."""
    if "curve" in row:
        pts = [(seconds_from_start(t, start_time), float(v)) for t, v in row["curve"]]
        if len(pts) < 2:
            raise ValueError(f"sources[{i}] curve needs at least two points")
        pts.sort()
        tp = np.array([p[0] for p in pts])
        vp = np.array([p[1] for p in pts])
        if np.any(vp < 0.0):
            raise ValueError(f"sources[{i}] curve values must be non-negative")
        outside = (tp < 0.0) | (tp > total)
        if np.any(vp[outside] > 0.0):
            warnings.warn(f"sources[{i}] curve extends outside the run window and is truncated", UserWarning, stacklevel=4)
        inner = tp[(tp > 0.0) & (tp < total)]
        t = np.unique(np.concatenate([[0.0, total], inner]))
        v = np.interp(t, tp, vp, left=0.0, right=0.0)
        return t, v
    value = float(row[key])
    if value < 0.0:
        raise ValueError(f"sources[{i}] {key} must be non-negative")
    t0 = seconds_from_start(row.get("start", 0.0), start_time)
    t1 = seconds_from_start(row.get("end", total), start_time)
    if t1 <= t0:
        raise ValueError(f"sources[{i}] end must be after start")
    if t0 < 0.0 or t1 > total:
        warnings.warn(f"sources[{i}] window is truncated to the run window", UserWarning, stacklevel=4)
    t0, t1 = max(t0, 0.0), min(t1, total)
    if t1 <= t0:
        raise ValueError(f"sources[{i}] lies entirely outside the run window")
    return np.array([t0, t1]), np.array([value, value])


def _cumulative(t: FloatArray, rate: FloatArray) -> FloatArray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])


def expand_sources(
    sources: Sequence[Mapping[str, Any]],
    network: Network,
    provider: HydraulicsProvider,
    *,
    start_time: np.datetime64,
    end_time: np.datetime64,
    particle_mass: float | None,
    rng: np.random.RandomState,
) -> ParticleSchedule:
    """Expand source rows into a particle schedule.

    Args:
        sources: validated source rows (see NetworkConfig).
        network: the network (for reach ids and lengths).
        provider: hydraulics, used for the flow of concentration-form sources.
        start_time: run start; source times are seconds from it or ISO datetimes.
        end_time: run end.
        particle_mass: global mass per particle; rows with ``particles`` override it.
        rng: random state for Poisson spacing.

    Returns:
        The concatenated schedule, in source order.

    Raises:
        ValueError: a source has no mass inside the run window, or an invalid position.
        KeyError: unknown reach id.
    """
    total = float((end_time - start_time) / np.timedelta64(1, "s"))
    parts: list[ParticleSchedule] = []
    for i, row in enumerate(sources):
        idx, s = _resolve_position(i, row, network)
        form = row["form"]
        if form == "slug":
            t_slug = seconds_from_start(row["time"], start_time)
            if not 0.0 <= t_slug <= total:
                raise ValueError(f"sources[{i}] slug time {t_slug} s is outside the run window [0, {total}]")
            mass_total = float(row["mass"])
            count, masses = _count_and_masses(i, row, mass_total, particle_mass, rng)
            times = np.full(count, t_slug)
        else:
            if form == "concentration":
                t, v = _concentration_rate(i, row, provider, start_time, idx, s / float(network.length[idx]), total)
            else:
                t, v = _curve(i, row, "rate", start_time, total)
            cum = _cumulative(t, v)
            mass_total = float(cum[-1])
            count, masses = _count_and_masses(i, row, mass_total, particle_mass, rng)
            times = _release_times(row, count, mass_total, cum, t, rng)
        parts.append(
            ParticleSchedule(
                np.full(count, idx, dtype=np.int32), np.full(count, s, dtype=np.float64), times.astype(np.float64),
                masses, np.full(count, i, dtype=np.int32),
            )
        )
    return ParticleSchedule.concat(parts)


def _count_and_masses(
    i: int, row: Mapping[str, Any], mass_total: float, particle_mass: float | None, rng: np.random.RandomState
) -> tuple[int, FloatArray]:
    if mass_total <= 0.0:
        raise ValueError(f"sources[{i}] has zero mass inside the run window")
    poisson = row.get("spacing", "even") == "poisson"
    if "particles" in row:
        count = int(row["particles"])
        return count, np.full(count, mass_total / count)
    assert particle_mass is not None
    if poisson:
        count = max(1, int(rng.poisson(mass_total / particle_mass)))
        return count, np.full(count, mass_total / count)
    count = max(1, int(round(mass_total / particle_mass)))
    masses = np.full(count, particle_mass)
    masses[-1] = mass_total - particle_mass * (count - 1)
    return count, masses


def _release_times(
    row: Mapping[str, Any], count: int, mass_total: float, cum: FloatArray, t: FloatArray, rng: np.random.RandomState
) -> FloatArray:
    spacing = row.get("spacing", "even")
    if spacing == "poisson":
        q = np.sort(rng.uniform(0.0, mass_total, count))
    elif spacing == "even":
        q = (np.arange(count) + 0.5) / count * mass_total
    else:
        raise ValueError(f"spacing must be 'even' or 'poisson', got {spacing!r}")
    keep = np.concatenate([[True], np.diff(cum) > 0.0])
    return np.interp(q, cum[keep], t[keep])


def _flow_at(provider: HydraulicsProvider, start_time: np.datetime64, idx: int, frac: float, t: FloatArray) -> FloatArray:
    """Flow at fraction ``frac`` along reach ``idx`` at seconds ``t`` (requests in increasing time)."""
    q = np.empty(t.size)
    for j, tj in enumerate(t):
        h = provider.hydraulics(start_time + np.timedelta64(int(round(tj * 1e9)), "ns"))
        q[j] = h["flow_in"][idx] + (h["flow_out"][idx] - h["flow_in"][idx]) * frac
    return q


def _concentration_rate(
    i: int, row: Mapping[str, Any], provider: HydraulicsProvider, start_time: np.datetime64, idx: int, frac: float, total: float
) -> tuple[FloatArray, FloatArray]:
    """Mass rate C(t) * Q(t) at the union of the curve's breakpoints and the provider's timestamps in between."""
    t_c, c = _curve(i, row, "value", start_time, total)
    ts = (provider.times - start_time) / np.timedelta64(1, "s")
    inner = ts[(ts > t_c[0]) & (ts < t_c[-1])]
    t_all = np.unique(np.concatenate([t_c, inner]))
    c_all = np.interp(t_all, t_c, c, left=0.0, right=0.0)
    return t_all, c_all * _flow_at(provider, start_time, idx, frac, t_all)
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_sources.py -v` → 11 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/sources.py tests/network/test_sources.py
git commit -m "Add mass-loading sources expanded into a particle release schedule

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Particle budget helper

**Files:**
- Modify: `src/fluvial_particle/network/sources.py` (append `estimate_particles`)
- Test: `tests/network/test_sources.py` (append)

**Interfaces:**
- Consumes: `NetworkConfig`, `HydraulicsProvider`, `dispersion_coefficient`.
- Produces: `estimate_particles(config, provider, *, target_per_bin=100.0, bin_length=100.0, reference_travel_time=86400.0) -> pandas.DataFrame` with columns `source`, `form`, `reach_id`, `total_mass`, `particle_mass`, `particles`.

- [ ] **Step 1: Append failing tests**

```python
from fluvial_particle.network.config import NetworkConfig
from fluvial_particle.network.sources import estimate_particles


def test_estimate_particles_hand_values(env):
    net, prov = env
    cfg = NetworkConfig.from_dict({
        "hydraulics_file": "unused.nc",
        "dispersion": {"model": "constant", "value": 10.0},
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 100.0, "particles": 1},
            {"reach_id": 101, "form": "loading", "rate": 0.01, "start": 0.0, "end": DAY, "particles": 1},
        ],
    })
    df = estimate_particles(cfg, prov, target_per_bin=100.0, bin_length=100.0, reference_travel_time=DAY)
    sigma = np.sqrt(2 * 10.0 * DAY)
    n_slug = int(np.ceil(100.0 * sigma * np.sqrt(2 * np.pi) / 100.0))
    assert df.loc[0, "particles"] == n_slug
    assert df.loc[0, "particle_mass"] == pytest.approx(100.0 / n_slug)
    # continuous: m = rate * bin / (v * target) = 0.01 * 100 / (1.0 * 100) = 0.01; N = M / m = 864 / 0.01
    assert df.loc[1, "particle_mass"] == pytest.approx(0.01)
    assert df.loc[1, "particles"] == 86400
    assert list(df.columns) == ["source", "form", "reach_id", "total_mass", "particle_mass", "particles"]
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_sources.py -k estimate -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Append to `sources.py`**

Add imports at the top: `import pandas as pd`, `from .config import NetworkConfig, parse_datetime`, `from .dispersion import dispersion_coefficient`.

```python
def estimate_particles(
    config: NetworkConfig,
    provider: HydraulicsProvider,
    *,
    target_per_bin: float = 100.0,
    bin_length: float = 100.0,
    reference_travel_time: float = 86400.0,
) -> pd.DataFrame:
    """Particle mass and count per source for about ``target_per_bin`` particles in a ``bin_length`` bin.

    Continuous sources: m = rate * bin_length / (v * target) with v the release reach's mean velocity over the
    source window. Slugs: the count that puts ``target`` particles in the peak bin after ``reference_travel_time``,
    N = target * sigma * sqrt(2 pi) / bin_length with sigma = sqrt(2 K t). The config is not changed.

    Returns:
        DataFrame with columns source, form, reach_id, total_mass, particle_mass, particles.
    """
    network = Network(provider.static, crs_wkt=provider.crs_wkt)
    start, end = config.resolve_times(provider.times)
    total = float((end - start) / np.timedelta64(1, "s"))
    disp = config.dispersion
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(config.sources):
        idx, s = _resolve_position(i, row, network)
        form = row["form"]
        if form == "slug":
            t0 = t1 = seconds_from_start(row["time"], start)
            mass_total = float(row["mass"])
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if form == "concentration":
                    t, v = _concentration_rate(i, row, provider, start, idx, s / float(network.length[idx]), total)
                else:
                    t, v = _curve(i, row, "rate", start, total)
            t0, t1 = float(t[0]), float(t[-1])
            mass_total = float(_cumulative(t, v)[-1])
        sample = np.unique(np.clip(np.linspace(t0, t1, 8), 0.0, total))
        vel: list[float] = []
        kk: list[float] = []
        for tj in sample:
            h = provider.hydraulics(start + np.timedelta64(int(round(tj * 1e9)), "ns"))
            vel.append(float(h["velocity"][idx]))
            kk.append(float(dispersion_coefficient(h, disp.model, scale=disp.scale, cap=disp.cap, value=disp.value)[idx]))
        v_mean = max(float(np.mean(vel)), 1e-12)
        k_mean = float(np.mean(kk))
        if form == "slug":
            sigma = np.sqrt(2.0 * k_mean * reference_travel_time)
            n = max(1, int(np.ceil(target_per_bin * max(sigma * np.sqrt(2.0 * np.pi) / bin_length, 1.0))))
            m = mass_total / n
        else:
            rate_mean = mass_total / max(t1 - t0, 1e-12)
            m = rate_mean * bin_length / (v_mean * target_per_bin)
            n = max(1, int(np.ceil(mass_total / m)))
        rows.append({"source": i, "form": form, "reach_id": int(row["reach_id"]), "total_mass": mass_total, "particle_mass": m, "particles": n})
    return pd.DataFrame(rows, columns=["source", "form", "reach_id", "total_mass", "particle_mass", "particles"])
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_sources.py -v` → 12 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/sources.py tests/network/test_sources.py
git commit -m "Add estimate_particles budget helper

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Solver: state, release, exact advection with time carry

**Files:**
- Create: `src/fluvial_particle/network/solver.py`
- Test: `tests/network/test_solver.py`

**Interfaces:**
- Consumes: `Network`, `HydraulicsProvider`, `ParticleSchedule`, `DispersionConfig`, `dispersion_coefficient`.
- Produces: `NetworkSolver(network, provider, schedule, *, start_time, dt, dispersion, rng, max_hops=1000)` with arrays `reach: NDArray[int32]`, `s: NDArray[floating]`, `prev_reach: NDArray[int32]`, `status: NDArray[int8]`, `mass`, `release_reach`, `release_s`, `release_time`, `exit_time: NDArray[float64]`, `exit_reach: NDArray[int32]`; `time: float`; `n: int`; `step() -> None`; `midpoint_time() -> np.datetime64`; `distance_along(cum_before) -> FloatArray` (helper for tests); constants `UNRELEASED, ACTIVE, EXITED = 0, 1, 2`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_solver.py`:

```python
"""Tests for NetworkSolver stepping."""

import numpy as np
import pytest

from fluvial_particle.network.config import DispersionConfig
from fluvial_particle.network.network import Network
from fluvial_particle.network.solver import ACTIVE, EXITED, UNRELEASED, NetworkSolver
from fluvial_particle.network.sources import ParticleSchedule
from tests.network.support import ArrayHydraulicsProvider, chain_dataset, three_reach_dataset

T0 = np.datetime64("1979-01-01", "ns")
NONE = DispersionConfig(model="none")


def make_solver(ds, schedule, dt, dispersion=NONE, seed=0, max_hops=1000, rng=None):
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    net = Network(prov.static)
    return NetworkSolver(
        net, prov, schedule, start_time=T0, dt=dt, dispersion=dispersion,
        rng=rng or np.random.RandomState(seed), max_hops=max_hops,
    )


def test_release_at_start_and_plain_advection():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 100.0, 0.0), dt=200.0)
    assert sol.status[0] == UNRELEASED and sol.reach[0] == -1 and np.isnan(sol.s[0])
    sol.step()
    assert sol.status[0] == ACTIVE and sol.reach[0] == 0
    assert sol.s[0] == pytest.approx(300.0)  # v = 1 m/s
    assert sol.time == 200.0
    assert sol.prev_reach[0] == -1


def test_one_hop_with_time_carry():
    # reach 0 (1000 m, 1 m/s) -> reach 2 (2 m/s): 900 + 200 overshoots by 100 m = 100 s left -> 200 m into reach 2
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 900.0, 0.0), dt=200.0)
    sol.step()
    assert sol.reach[0] == 2 and sol.s[0] == pytest.approx(200.0) and sol.prev_reach[0] == 0


def test_two_hops_in_one_step():
    ds = chain_dataset(n_reach=4, length=1000.0, velocity=[1.0, 2.0, 4.0, 1.0])
    sol = make_solver(ds, ParticleSchedule.simple(0, 0.0, 0.0), dt=1600.0)  # 1000 s + 500 s + 100 s at 4 m/s
    sol.step()
    assert sol.reach[0] == 2 and sol.s[0] == pytest.approx(400.0) and sol.prev_reach[0] == 1


def test_hop_into_zero_velocity_reach_waits():
    ds = three_reach_dataset(velocity=[1.0, 0.5, 0.0], flow_out=[10.0, 5.0, 0.0])
    sol = make_solver(ds, ParticleSchedule.simple(0, 900.0, 0.0), dt=200.0)
    sol.step()
    assert sol.reach[0] == 2 and sol.s[0] == 0.0 and sol.status[0] == ACTIVE
    sol.step()
    assert sol.s[0] == 0.0


def test_exit_records_exact_time():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(2, 2900.0, 0.0), dt=100.0)
    sol.step()  # 2 m/s: reaches the end after 50 s
    assert sol.status[0] == EXITED and sol.reach[0] == -1 and np.isnan(sol.s[0])
    assert sol.exit_time[0] == pytest.approx(50.0) and sol.exit_reach[0] == 2
    sol.step()  # exited particles stay put
    assert sol.exit_time[0] == pytest.approx(50.0)


def test_mid_step_release_uses_partial_time():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 0.0, 50.0), dt=100.0)
    sol.step()
    assert sol.status[0] == ACTIVE and sol.s[0] == pytest.approx(50.0)
    later = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 0.0, 250.0), dt=100.0)
    later.step()
    later.step()
    assert later.status[0] == UNRELEASED
    later.step()
    assert later.status[0] == ACTIVE and later.s[0] == pytest.approx(50.0)


def test_mass_conservation_and_vectorized_batch():
    n = 50
    sch = ParticleSchedule(
        np.zeros(n, dtype=np.int32), np.linspace(0.0, 1000.0, n), np.linspace(0.0, 400.0, n),
        np.ones(n), np.zeros(n, dtype=np.int32),
    )
    sol = make_solver(three_reach_dataset(), sch, dt=300.0)
    for _ in range(8):  # 2400 s: the earliest particles have exited, the latest are still in reach 2
        sol.step()
        counts = np.bincount(sol.status, minlength=3)
        assert counts.sum() == n
    assert (sol.status == EXITED).sum() > 0 and (sol.status == ACTIVE).sum() > 0


def test_distance_along_helper():
    ds = chain_dataset(n_reach=3, length=1000.0, velocity=1.0)
    sol = make_solver(ds, ParticleSchedule.simple(0, 0.0, 0.0), dt=1500.0)
    sol.step()
    cum_before = np.array([0.0, 1000.0, 2000.0])
    assert sol.distance_along(cum_before)[0] == pytest.approx(1500.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_solver.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `solver.py` (advection; `_disperse` is a no-op stub replaced in Task 11)**

```python
"""Vectorized 1D network particle solver: exact advection with time carry, then one dispersive kick."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .config import DispersionConfig
from .dispersion import dispersion_coefficient
from .network import Network
from .provider import HydraulicsProvider
from .sources import ParticleSchedule


UNRELEASED, ACTIVE, EXITED = 0, 1, 2
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class NetworkSolver:
    """Particle state arrays and the per-step update.

    Args:
        network: static topology.
        provider: per-step hydraulics.
        schedule: this solver's particles (already sliced for the MPI rank).
        start_time: run start; the solver clock is seconds from it.
        dt: step (s).
        dispersion: dispersion settings.
        rng: random state supplying standard normals.
        max_hops: cap on reach hops per particle per step; exceeding it raises RuntimeError.
    """

    def __init__(
        self,
        network: Network,
        provider: HydraulicsProvider,
        schedule: ParticleSchedule,
        *,
        start_time: np.datetime64,
        dt: float,
        dispersion: DispersionConfig,
        rng: np.random.RandomState,
        max_hops: int = 1000,
    ) -> None:
        self.network = network
        self.provider = provider
        self.start_time = np.datetime64(start_time, "ns")
        self.dt = float(dt)
        self.dispersion = dispersion
        self.rng = rng
        self.max_hops = int(max_hops)
        self.time = 0.0
        n = schedule.n
        self.n = n
        fdtype = provider.dtype
        self.release_reach = schedule.release_reach.astype(np.int32)
        self.release_s = schedule.release_s.astype(np.float64)
        self.release_time = schedule.release_time.astype(np.float64)
        self.mass = schedule.mass.astype(np.float64)
        self.source_index = schedule.source_index.astype(np.int32)
        self.reach: npt.NDArray[np.int32] = np.full(n, -1, dtype=np.int32)
        self.s: npt.NDArray[Any] = np.full(n, np.nan, dtype=fdtype)
        self.prev_reach: npt.NDArray[np.int32] = np.full(n, -1, dtype=np.int32)
        self.status: npt.NDArray[np.int8] = np.full(n, UNRELEASED, dtype=np.int8)
        self.exit_time: FloatArray = np.full(n, np.nan)
        self.exit_reach: npt.NDArray[np.int32] = np.full(n, -1, dtype=np.int32)
        self._length = network.length
        self._to_index = network.to_index.astype(np.int64)

    def midpoint_time(self) -> np.datetime64:
        """Datetime at the middle of the step about to be taken."""
        return self.start_time + np.timedelta64(int(round((self.time + 0.5 * self.dt) * 1e9)), "ns")

    def distance_along(self, cum_before: FloatArray) -> FloatArray:
        """Distance from the network origin for active particles: cum_before[reach] + s (NaN otherwise)."""
        out = np.full(self.n, np.nan)
        act = self.status == ACTIVE
        out[act] = cum_before[self.reach[act]] + self.s[act]
        return out

    def step(self) -> None:
        """Advance the clock by dt: release, advect with time carry, disperse with displacement carry."""
        t = self.time
        dt = self.dt
        tau = self._release(t, dt)
        h = self.provider.hydraulics(self.midpoint_time())
        v = np.asarray(h["velocity"], dtype=np.float64)
        d = self.dispersion
        k = dispersion_coefficient(h, d.model, scale=d.scale, cap=d.cap, value=d.value)
        self._advect(v, tau, t, dt)
        self._disperse(k, tau, t, dt)
        self.time = t + dt

    def _release(self, t: float, dt: float) -> FloatArray:
        """Activate particles due in (t, t + dt] (and any still pending at t); return per-particle time budgets."""
        tau = np.full(self.n, dt)
        new = (self.status == UNRELEASED) & (self.release_time <= t + dt)
        if new.any():
            self.status[new] = ACTIVE
            self.reach[new] = self.release_reach[new]
            self.s[new] = self.release_s[new]
            self.prev_reach[new] = -1
            tau[new] = t + dt - np.maximum(self.release_time[new], t)
        return tau

    def _advect(self, v: FloatArray, tau: FloatArray, t: float, dt: float) -> None:
        idx = np.nonzero(self.status == ACTIVE)[0]
        if idx.size == 0:
            return
        self.s[idx] += v[self.reach[idx]] * tau[idx]
        for _ in range(self.max_hops):
            over = self.s[idx] > self._length[self.reach[idx]]
            if not over.any():
                return
            j = idx[over]
            rj = self.reach[j].astype(np.int64)
            time_left = (self.s[j] - self._length[rj]) / v[rj]
            self.prev_reach[j] = rj
            nxt = self._to_index[rj]
            exiting = nxt < 0
            e = j[exiting]
            self.status[e] = EXITED
            self.exit_time[e] = t + dt - time_left[exiting]
            self.exit_reach[e] = rj[exiting]
            self.reach[e] = -1
            self.s[e] = np.nan
            m = j[~exiting]
            self.reach[m] = nxt[~exiting]
            self.s[m] = v[nxt[~exiting]] * time_left[~exiting]
            idx = m
        raise RuntimeError(f"a particle hopped more than max_hops={self.max_hops} reaches in one advection step")

    def _disperse(self, k: FloatArray, tau: FloatArray, t: float, dt: float) -> None:
        """Replaced in the dispersion task."""
        return
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_solver.py -v` → 8 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/solver.py tests/network/test_solver.py
git commit -m "Add NetworkSolver with release and exact advection across reach boundaries

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Solver: dispersive kick with displacement carry

**Files:**
- Modify: `src/fluvial_particle/network/solver.py` (replace `_disperse`)
- Test: `tests/network/test_solver.py` (append)

**Interfaces:**
- Produces: `NetworkSolver._disperse(k, tau, t, dt)`: one `rng.standard_normal` draw per active particle; downstream overshoot hops with `s = overshoot`, upstream overshoot returns to `prev_reach` (then `prev_reach = -1`) or reflects; exits set `exit_time = t + dt`.

- [ ] **Step 1: Append failing tests**

```python
class FixedNormals:
    """rng stub returning preset standard normals."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def standard_normal(self, n):
        assert n == self.values.size
        return self.values.copy()


K50 = DispersionConfig(model="constant", value=50.0)
STILL = {"velocity": [0.0, 0.0, 0.0], "flow_out": [10.0, 5.0, 80.0]}  # flow > 0 so K applies, but no advection


def test_kick_scale_is_sqrt_2k_tau():
    # sqrt(2 * 50 * 100) = 100 m per unit normal
    sol = make_solver(three_reach_dataset(**STILL), ParticleSchedule.simple(0, 500.0, 0.0), dt=100.0, dispersion=K50, rng=FixedNormals([0.5]))
    sol.step()
    assert sol.s[0] == pytest.approx(550.0) and sol.reach[0] == 0


def test_dispersive_downstream_hop_carries_displacement():
    sol = make_solver(three_reach_dataset(**STILL), ParticleSchedule.simple(0, 900.0, 0.0), dt=100.0, dispersion=K50, rng=FixedNormals([2.0]))
    sol.step()
    assert sol.reach[0] == 2 and sol.s[0] == pytest.approx(100.0) and sol.prev_reach[0] == 0


def test_dispersive_exit_at_end_of_step():
    sol = make_solver(three_reach_dataset(**STILL), ParticleSchedule.simple(2, 2950.0, 0.0), dt=100.0, dispersion=K50, rng=FixedNormals([1.0]))
    sol.step()
    assert sol.status[0] == EXITED and sol.exit_time[0] == 100.0 and sol.exit_reach[0] == 2


def test_upstream_hop_returns_to_previous_reach():
    sol = make_solver(three_reach_dataset(**STILL), ParticleSchedule.simple(2, 50.0, 0.0), dt=100.0, dispersion=K50, rng=FixedNormals([-1.0]))
    sol.status[:] = ACTIVE  # pre-place the particle with history
    sol.reach[0], sol.s[0], sol.prev_reach[0] = 2, 50.0, 0
    sol.release_time[0] = -1.0  # already released
    sol.step()
    assert sol.reach[0] == 0 and sol.s[0] == pytest.approx(950.0) and sol.prev_reach[0] == -1


def test_reflect_without_history_and_after_second_overshoot():
    sol = make_solver(three_reach_dataset(**STILL), ParticleSchedule.simple(0, 50.0, 0.0), dt=100.0, dispersion=K50, rng=FixedNormals([-1.0]))
    sol.step()
    assert sol.reach[0] == 0 and sol.s[0] == pytest.approx(50.0)
    sol = make_solver(three_reach_dataset(**STILL), ParticleSchedule.simple(2, 50.0, 0.0), dt=100.0, dispersion=K50, rng=FixedNormals([-11.0]))
    sol.status[:] = ACTIVE
    sol.reach[0], sol.s[0], sol.prev_reach[0] = 2, 50.0, 0
    sol.release_time[0] = -1.0
    sol.step()  # -1050: back into reach 0 at -50, then reflect to 50
    assert sol.reach[0] == 0 and sol.s[0] == pytest.approx(50.0) and sol.prev_reach[0] == -1


def test_max_hops_raises_on_cycle():
    prov = ArrayHydraulicsProvider.from_dataset(three_reach_dataset(velocity=[100.0, 100.0, 100.0]))
    static = dict(prov.static)
    static["to_index"] = np.array([1, 0, -1], dtype=np.int32)  # 0 <-> 1 cycle, never validated here
    net = Network(static)
    sol = NetworkSolver(net, prov, ParticleSchedule.simple(0, 0.0, 0.0), start_time=T0, dt=1000.0, dispersion=NONE, rng=np.random.RandomState(0), max_hops=3)
    with pytest.raises(RuntimeError, match="max_hops"):
        sol.step()


def test_seeded_reproducibility_and_conservation_with_dispersion():
    n = 200
    sch = ParticleSchedule(np.zeros(n, dtype=np.int32), np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n, dtype=np.int32))
    a = make_solver(three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    b = make_solver(three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    for _ in range(12):
        a.step()
        b.step()
        assert np.bincount(a.status, minlength=3).sum() == n
    np.testing.assert_array_equal(a.reach, b.reach)
    np.testing.assert_allclose(a.s, b.s, equal_nan=True)
    assert np.all((a.s[a.status == ACTIVE] >= 0.0) & (a.s[a.status == ACTIVE] <= a.network.length[a.reach[a.status == ACTIVE]]))
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_solver.py -k "kick or dispersive or upstream or reflect or max_hops or seeded" -v`
Expected: FAIL (kicks have no effect; `max_hops` test fails because the cycle test only exercises advection — that one passes already; the rest fail)

- [ ] **Step 3: Replace `_disperse` in `solver.py`**

```python
    def _disperse(self, k: FloatArray, tau: FloatArray, t: float, dt: float) -> None:
        idx = np.nonzero(self.status == ACTIVE)[0]
        if idx.size == 0:
            return
        xi = np.asarray(self.rng.standard_normal(idx.size), dtype=np.float64)
        self.s[idx] += xi * np.sqrt(2.0 * k[self.reach[idx]] * tau[idx])
        for _ in range(self.max_hops):
            r = self.reach[idx].astype(np.int64)
            down = self.s[idx] > self._length[r]
            up = self.s[idx] < 0.0
            if not (down.any() or up.any()):
                return
            # downstream: carry the overshoot into the next reach, or exit at the end of the step
            j = idx[down]
            rj = self.reach[j].astype(np.int64)
            over = self.s[j] - self._length[rj]
            self.prev_reach[j] = rj
            nxt = self._to_index[rj]
            exiting = nxt < 0
            e = j[exiting]
            self.status[e] = EXITED
            self.exit_time[e] = t + dt
            self.exit_reach[e] = rj[exiting]
            self.reach[e] = -1
            self.s[e] = np.nan
            m = j[~exiting]
            self.reach[m] = nxt[~exiting]
            self.s[m] = over[~exiting]
            # upstream: back into the reach the particle came from, else reflect
            u = idx[up]
            pr = self.prev_reach[u].astype(np.int64)
            has = pr >= 0
            a = u[has]
            self.s[a] = self._length[pr[has]] + self.s[a]
            self.reach[a] = pr[has]
            self.prev_reach[a] = -1
            b = u[~has]
            self.s[b] = -self.s[b]
            idx = np.concatenate([m, u])
        raise RuntimeError(f"a particle hopped more than max_hops={self.max_hops} reaches in one dispersion step")
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_solver.py -v` → 15 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/solver.py tests/network/test_solver.py
git commit -m "Add dispersive kick with displacement carry and upstream history hops

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: NetCDF writer

**Files:**
- Create: `src/fluvial_particle/network/writer.py`
- Test: `tests/network/test_writer.py`

**Interfaces:**
- Consumes: `ParticleSchedule`.
- Produces: `NetworkWriter(path, *, n_particles, reach_id, start_time, attrs, dtype=np.float64, comm=None)` with `write_schedule(schedule, lo, hi)`, `write_step(itime, time_seconds, reach, s, status, lo, hi)`, `write_exits(exit_time, exit_reach, lo, hi)`, `close()`, context manager; constant `OUTPUT_FILENAME = "network_particles.nc"`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_writer.py`:

```python
"""Tests for NetworkWriter round trips."""

import json

import numpy as np
import xarray as xr

from fluvial_particle.network.sources import ParticleSchedule
from fluvial_particle.network.writer import OUTPUT_FILENAME, NetworkWriter

T0 = np.datetime64("1979-01-01", "ns")


def test_round_trip(tmp_path):
    path = tmp_path / OUTPUT_FILENAME
    sch = ParticleSchedule(
        np.array([0, 0, 1], dtype=np.int32), np.array([0.0, 5.0, 10.0]), np.array([0.0, 0.0, 30.0]),
        np.array([1.0, 1.0, 2.0]), np.array([0, 0, 1], dtype=np.int32),
    )
    attrs = {"dt": 60.0, "sources": json.dumps([{"reach_id": 1}]), "mass_units": "kg"}
    with NetworkWriter(path, n_particles=3, reach_id=np.array([101, 102]), start_time=T0, attrs=attrs) as w:
        w.write_schedule(sch, 0, 3)
        w.write_step(0, 0.0, np.array([0, 0, -1], dtype=np.int32), np.array([0.0, 5.0, np.nan]), np.array([1, 1, 0], dtype=np.int8), 0, 3)
        w.write_step(1, 60.0, np.array([-1, 0, 1], dtype=np.int32), np.array([np.nan, 65.0, 40.0]), np.array([2, 1, 1], dtype=np.int8), 0, 3)
        w.write_exits(np.array([42.0, np.nan, np.nan]), np.array([1, -1, -1], dtype=np.int32), 0, 3)
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds.sizes == {"time": 2, "particle": 3, "reach": 2}
        assert ds["time"].values[1] == T0 + np.timedelta64(60, "s")
        assert list(ds["time_seconds"].values) == [0.0, 60.0]
        assert ds["reach_index"].dtype == np.int32 and list(ds["reach_index"].values[1]) == [-1, 0, 1]
        assert np.isnan(ds["s"].values[0, 2]) and ds["s"].values[1, 1] == 65.0
        assert list(ds["status"].values[0]) == [1, 1, 0]
        assert list(ds["mass"].values) == [1.0, 1.0, 2.0]
        assert list(ds["release_time"].values) == [0.0, 0.0, 30.0]
        assert ds["exit_time"].values[0] == 42.0 and np.isnan(ds["exit_time"].values[1])
        assert list(ds["exit_reach"].values) == [1, -1, -1]
        assert list(ds["reach_id"].values) == [101, 102]
        assert ds.attrs["dt"] == 60.0 and json.loads(ds.attrs["sources"])[0]["reach_id"] == 1
        assert ds["reach_index"].encoding.get("chunksizes") == (1, 3)


def test_partial_slices_and_dtype(tmp_path):
    path = tmp_path / OUTPUT_FILENAME
    sch = ParticleSchedule(np.zeros(4, dtype=np.int32), np.zeros(4), np.zeros(4), np.ones(4), np.zeros(4, dtype=np.int32))
    with NetworkWriter(path, n_particles=4, reach_id=np.array([1]), start_time=T0, attrs={}, dtype=np.float32) as w:
        w.write_schedule(sch.slice(0, 2), 0, 2)
        w.write_schedule(sch.slice(2, 4), 2, 4)
        w.write_step(0, 0.0, np.zeros(2, dtype=np.int32), np.full(2, 1.5, dtype=np.float32), np.ones(2, dtype=np.int8), 0, 2)
        w.write_step(0, 0.0, np.zeros(2, dtype=np.int32), np.full(2, 2.5, dtype=np.float32), np.ones(2, dtype=np.int8), 2, 4)
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds["s"].dtype == np.float32
        assert list(ds["s"].values[0]) == [1.5, 1.5, 2.5, 2.5]
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_writer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `writer.py`**

```python
"""NetCDF4 particle output through h5netcdf on h5py, serial or with the mpio driver."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import h5netcdf
import numpy as np
import numpy.typing as npt

from .sources import ParticleSchedule


OUTPUT_FILENAME = "network_particles.nc"
MAX_PARTICLE_CHUNK = 2**18


class NetworkWriter:
    """Write the network particle file one output time at a time.

    Every rank constructs the writer (dataset creation is collective under mpio); each rank writes its own
    particle slice ``lo:hi``. The ``time`` dimension is unlimited and resized collectively in write_step.

    Args:
        path: output file.
        n_particles: total particle count across ranks.
        reach_id: ids of the run's reaches (subset order).
        start_time: run start; ``time`` is stored as seconds since it.
        attrs: global attributes (strings, numbers, or None; None is skipped).
        dtype: dtype of ``s``.
        comm: MPI communicator for parallel writes, or None.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        n_particles: int,
        reach_id: npt.NDArray[np.integer[Any]],
        start_time: np.datetime64,
        attrs: Mapping[str, Any],
        dtype: npt.DTypeLike = np.float64,
        comm: Any = None,
    ) -> None:
        self.path = pathlib.Path(path)
        kwargs: dict[str, Any] = {"driver": "mpio", "comm": comm} if comm is not None else {}
        self._f = h5netcdf.File(str(self.path), "w", **kwargs)
        n = int(n_particles)
        self._f.dimensions = {"time": None, "particle": n, "reach": int(reach_id.size)}
        chunk = (1, max(1, min(n, MAX_PARTICLE_CHUNK)))
        f = self._f
        start_iso = str(np.datetime64(start_time, "s"))
        v = f.create_variable("time", ("time",), dtype="f8", chunks=(1024,))
        v.attrs["units"] = f"seconds since {start_iso}"
        v.attrs["calendar"] = "proleptic_gregorian"
        v.attrs["long_name"] = "output time"
        v = f.create_variable("time_seconds", ("time",), dtype="f8", chunks=(1024,))
        v.attrs["units"] = "s"
        v.attrs["long_name"] = "seconds since start_time"
        v = f.create_variable("reach_index", ("time", "particle"), dtype="i4", chunks=chunk, fillvalue=-1)
        v.attrs["long_name"] = "reach index of the particle, -1 when not active"
        v = f.create_variable("s", ("time", "particle"), dtype=np.dtype(dtype), chunks=chunk, fillvalue=np.nan)
        v.attrs["units"] = "m"
        v.attrs["long_name"] = "distance from the reach's upstream end, NaN when not active"
        v = f.create_variable("status", ("time", "particle"), dtype="i1", chunks=chunk, fillvalue=0)
        v.attrs["long_name"] = "0 unreleased, 1 active, 2 exited"
        for name, dt_, units, long_name in (
            ("mass", "f8", None, "particle mass"),
            ("source_index", "i4", None, "row in the source table"),
            ("release_reach", "i4", None, "release reach index"),
            ("release_s", "f8", "m", "release distance from the reach's upstream end"),
            ("release_time", "f8", "s", "release time, seconds since start_time"),
            ("exit_time", "f8", "s", "exit time, seconds since start_time; NaN until exit"),
            ("exit_reach", "i4", None, "outlet reach index at exit, -1 until exit"),
        ):
            fill: Any = np.nan if dt_ == "f8" and name == "exit_time" else (-1 if dt_ == "i4" and name == "exit_reach" else None)
            v = f.create_variable(name, ("particle",), dtype=dt_, fillvalue=fill)
            if units:
                v.attrs["units"] = units
            v.attrs["long_name"] = long_name
        v = f.create_variable("reach_id", ("reach",), dtype="i8", data=np.asarray(reach_id, dtype=np.int64))
        v.attrs["long_name"] = "reach ids in the run's reach order"
        f.attrs["start_time"] = start_iso
        for key, value in attrs.items():
            if value is not None:
                f.attrs[key] = value
        self._n_time = 0

    def __enter__(self) -> NetworkWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write_schedule(self, schedule: ParticleSchedule, lo: int, hi: int) -> None:
        """Write the per-particle release arrays for particles lo..hi-1."""
        f = self._f
        f.variables["mass"][lo:hi] = schedule.mass
        f.variables["source_index"][lo:hi] = schedule.source_index
        f.variables["release_reach"][lo:hi] = schedule.release_reach
        f.variables["release_s"][lo:hi] = schedule.release_s
        f.variables["release_time"][lo:hi] = schedule.release_time

    def write_step(
        self,
        itime: int,
        time_seconds: float,
        reach: npt.NDArray[np.integer[Any]],
        s: npt.NDArray[np.floating[Any]],
        status: npt.NDArray[np.integer[Any]],
        lo: int,
        hi: int,
    ) -> None:
        """Write one output time for particles lo..hi-1 (resizes ``time`` when itime is new; collective)."""
        f = self._f
        if itime >= self._n_time:
            f.resize_dimension("time", itime + 1)
            self._n_time = itime + 1
        f.variables["time"][itime] = float(time_seconds)
        f.variables["time_seconds"][itime] = float(time_seconds)
        f.variables["reach_index"][itime, lo:hi] = np.asarray(reach, dtype=np.int32)
        f.variables["s"][itime, lo:hi] = s
        f.variables["status"][itime, lo:hi] = np.asarray(status, dtype=np.int8)

    def write_exits(self, exit_time: npt.NDArray[np.floating[Any]], exit_reach: npt.NDArray[np.integer[Any]], lo: int, hi: int) -> None:
        """Write exit times and reaches for particles lo..hi-1 (called once at the end of the run)."""
        self._f.variables["exit_time"][lo:hi] = exit_time
        self._f.variables["exit_reach"][lo:hi] = np.asarray(exit_reach, dtype=np.int32)

    def close(self) -> None:
        """Flush and close the file."""
        self._f.close()
```

If h5netcdf rejects `fillvalue=None` for a variable, pass `fillvalue` only when it is not None (build the kwargs dict conditionally). If `resize_dimension` does not exist in the installed h5netcdf (1.8.1 has it), upgrade the floor in `pyproject.toml` to the first version that does.

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_writer.py -v` → 2 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/writer.py tests/network/test_writer.py
git commit -m "Add NetworkWriter for NetCDF4 particle output through h5netcdf

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 13: Run orchestration and startup diagnostics

**Files:**
- Create: `src/fluvial_particle/network/run.py`
- Create: `src/fluvial_particle/network/results.py` (minimal shell: constructor, `close`, context manager, `times`, `n_particles`; the rest is Task 14)
- Test: `tests/network/test_run.py`

**Interfaces:**
- Consumes: everything from Tasks 5-12.
- Produces: `run_network_simulation(config, output_dir, *, seed=None, comm=None, quiet=False) -> NetworkResults | None`; `diagnostics_report(network, provider, schedule, config, start, end) -> str`; `resolve_seed(seed, comm) -> int`; `NetworkResults(path)` shell with `.path`, `.times`, `.n_particles`, `.close()`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_run.py`:

```python
"""End-to-end tests for run_network_simulation."""

import numpy as np
import xarray as xr

from fluvial_particle.network.run import diagnostics_report, resolve_seed, run_network_simulation
from fluvial_particle.network.writer import OUTPUT_FILENAME
from tests.network.support import three_reach_dataset, write_network_file


def config_for(path, **extra):
    cfg = {
        "hydraulics_file": str(path),
        "dt": 600.0,
        "output_interval": 600.0,
        "end_time": "1979-01-01T02:00",
        "dispersion": {"model": "none"},
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 3.0, "particles": 3},
            {"reach_id": 102, "form": "loading", "rate": 0.001, "start": 0.0, "end": 3600.0, "particles": 2},
        ],
    }
    cfg.update(extra)
    return cfg


def test_run_writes_file_and_conserves_mass(tmp_path):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    res = run_network_simulation(config_for(path), tmp_path / "out", seed=1, quiet=True)
    assert (tmp_path / "out" / OUTPUT_FILENAME).exists()
    assert res.n_particles == 5
    assert res.times.size == 13  # t = 0 plus 12 hourly-tenth outputs over 2 h at 600 s
    with xr.open_dataset(tmp_path / "out" / OUTPUT_FILENAME, engine="h5netcdf") as ds:
        status = ds["status"].values
        assert all(np.bincount(row, minlength=3).sum() == 5 for row in status)  # every particle accounted for
        assert list(np.bincount(status[-1], minlength=3)) == [0, 2, 3]  # slug exited, loading particles still moving
        # reach 101 particles: 1000 m at 1 m/s then 3000 m at 2 m/s = 2500 s
        np.testing.assert_allclose(ds["exit_time"].values[:3], 2500.0)
        assert list(ds["exit_reach"].values[:3]) == [2, 2, 2]
        assert ds.attrs["seed"] == 1 and ds.attrs["interpolation"] == "linear"
        assert ds.attrs["hydraulics_file"].endswith("net.nc")
        assert "sources" in ds.attrs and "dispersion" in ds.attrs
        assert ds["reach_index"].values[1, 0] == 0 and ds["s"].values[1, 0] == 600.0
    res.close()


def test_run_accepts_toml_and_prints_report(tmp_path, capsys):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    toml = tmp_path / "run.toml"
    lines = ["[network]", f'hydraulics_file = "{path}"', "dt = 600.0", "output_interval = 1200.0",
             'end_time = "1979-01-01T01:00"', "particle_mass = 1.0",
             "[network.dispersion]", 'model = "none"',
             "[[network.sources]]", "reach_id = 101", 'form = "slug"', "time = 0.0", "mass = 2.0"]
    toml.write_text("\n".join(lines) + "\n")
    res = run_network_simulation(toml, tmp_path / "out2")
    out = capsys.readouterr().out
    assert "reaches" in out and "dispersion kick" in out and "memory" in out.lower()
    assert res.times.size == 4
    res.close()


def test_resolve_seed():
    assert resolve_seed(5, None) == 5
    s = resolve_seed(None, None)
    assert isinstance(s, int) and s >= 0
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the `results.py` shell**

```python
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
        p = pathlib.Path(path)
        self.path = p / OUTPUT_FILENAME if p.is_dir() else p
        self._ds = xr.open_dataset(self.path, engine="h5netcdf")

    def __enter__(self) -> NetworkResults:
        return self

    def __exit__(self, *exc: object) -> None:
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
```

- [ ] **Step 4: Implement `run.py`**

```python
"""Run a network particle simulation end to end."""

from __future__ import annotations

import json
import pathlib
import time as _time
from collections.abc import Mapping
from os import getpid
from typing import Any

import numpy as np

from .. import __version__
from .config import NetworkConfig
from .dispersion import dispersion_coefficient
from .network import Network
from .provider import FileHydraulicsProvider
from .results import NetworkResults
from .solver import NetworkSolver
from .sources import ParticleSchedule, expand_sources
from .writer import OUTPUT_FILENAME, NetworkWriter


def resolve_seed(seed: int | None, comm: Any) -> int:
    """A base seed shared by all ranks: the given one, or one derived from time and pid on rank 0."""
    if seed is None:
        seed = int(abs(((_time.time() * 181) * ((getpid() - 83) * 359)) % 104729))
    if comm is not None:
        seed = comm.bcast(int(seed), root=0)
    return int(seed)


def diagnostics_report(
    network: Network,
    provider: FileHydraulicsProvider,
    schedule: ParticleSchedule,
    config: NetworkConfig,
    start: np.datetime64,
    end: np.datetime64,
) -> str:
    """Human-readable startup summary: network, sources, dt consequences, memory."""
    lines = [
        "Network particle simulation",
        f"  reaches: {network.n_reach}, outlets: {network.outlets().size}, headwaters: {network.headwaters().size}, "
        f"total length: {network.length.sum() / 1000.0:.1f} km",
        f"  hydraulics: {provider.path} ({provider.times[0]} .. {provider.times[-1]}, {provider.times.size} steps, "
        f"{provider.interpolation})",
        f"  run: {start} .. {end}, dt = {config.dt} s, output every {config.output_interval} s",
        f"  sources: {len(config.sources)} rows, {schedule.n} particles, total mass {schedule.mass.sum():.6g} "
        f"{config.mass_units}",
    ]
    inside = provider.times[(provider.times >= start) & (provider.times <= end)]
    if inside.size == 0:
        inside = provider.times[[int(np.searchsorted(provider.times, start, side="right")) - 1]]
    sample = inside[np.unique(np.linspace(0, inside.size - 1, min(inside.size, 64)).astype(int))]
    kicks: list[np.ndarray] = []
    crossed: list[np.ndarray] = []
    d = config.dispersion
    for t in sample:
        h = provider.hydraulics(t)
        wet = np.asarray(h["flow_out"]) > 0.0
        k = dispersion_coefficient(h, d.model, scale=d.scale, cap=d.cap, value=d.value)
        kicks.append(np.sqrt(2.0 * k[wet] * config.dt))
        crossed.append(np.asarray(h["velocity"])[wet] * config.dt > network.length[wet])
    kick = np.concatenate(kicks) if kicks else np.zeros(0)
    cross = np.concatenate(crossed) if crossed else np.zeros(0, dtype=bool)
    med_len = float(np.median(network.length))
    if kick.size:
        lines.append(
            f"  dt check: dispersion kick median {np.median(kick):.0f} m, p95 {np.percentile(kick, 95):.0f} m "
            f"vs median reach length {med_len:.0f} m; {100.0 * cross.mean():.1f}% of wet reach-days crossed in one step"
        )
    est = provider.memory_estimate(schedule.n)
    lines.append(
        f"  memory: window {est['window_bytes'] / 1e6:.1f} MB, static {est['static_bytes'] / 1e6:.1f} MB, "
        f"particles {est['particle_bytes'] / 1e6:.1f} MB, per output time {est['output_time_bytes'] / 1e6:.1f} MB"
    )
    return "\n".join(lines)


def run_network_simulation(
    config: NetworkConfig | Mapping[str, Any] | str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    seed: int | None = None,
    comm: Any = None,
    quiet: bool = False,
) -> NetworkResults | None:
    """Run a network particle simulation and return the results (rank 0; other ranks return None).

    Args:
        config: NetworkConfig, a dict for NetworkConfig.from_dict, or a TOML path.
        output_dir: directory for ``network_particles.nc`` (created if missing).
        seed: base random seed; overrides config.seed. Rank r draws from seed + 1 + r.
        comm: MPI communicator for parallel runs.
        quiet: suppress the startup report.

    Returns:
        NetworkResults on rank 0, None elsewhere.
    """
    cfg = NetworkConfig.coerce(config)
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1
    out = pathlib.Path(output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    provider = FileHydraulicsProvider(
        cfg.hydraulics_file, interpolation=cfg.interpolation, dtype=cfg.dtype, reach_subset=cfg.reach_subset
    )
    start, end = cfg.resolve_times(provider.times)
    provider.time_window = (start, end)
    network = Network(provider.static, crs_wkt=provider.crs_wkt)
    base_seed = resolve_seed(seed if seed is not None else cfg.seed, comm)
    schedule = expand_sources(
        cfg.sources, network, provider, start_time=start, end_time=end,
        particle_mass=cfg.particle_mass, rng=np.random.RandomState(base_seed),
    )
    n = schedule.n
    lo, hi = rank * n // size, (rank + 1) * n // size
    solver = NetworkSolver(
        network, provider, schedule.slice(lo, hi), start_time=start, dt=cfg.dt, dispersion=cfg.dispersion,
        rng=np.random.RandomState(base_seed + 1 + rank), max_hops=cfg.max_hops,
    )
    if rank == 0 and not quiet:
        print(diagnostics_report(network, provider, schedule, cfg, start, end), flush=True)

    attrs: dict[str, Any] = {
        "hydraulics_file": str(pathlib.Path(cfg.hydraulics_file).resolve()),
        "reach_subset": json.dumps(cfg.to_dict()["reach_subset"]),
        "interpolation": cfg.interpolation,
        "dt": cfg.dt,
        "output_interval": cfg.output_interval,
        "end_time": str(np.datetime64(end, "s")),
        "seed": base_seed,
        "mass_units": cfg.mass_units,
        "dispersion": json.dumps(cfg.dispersion.to_dict()),
        "sources": json.dumps(cfg.to_dict()["sources"]),
        "fluvial_particle_version": __version__,
        "created": str(np.datetime64("now", "s")),
        "conventions_note": provider.conventions_note,
    }
    total = float((end - start) / np.timedelta64(1, "s"))
    n_steps = int(np.floor(total / cfg.dt + 1e-9))
    every = int(round(cfg.output_interval / cfg.dt))
    with NetworkWriter(
        out / OUTPUT_FILENAME, n_particles=n, reach_id=network.reach_id, start_time=start, attrs=attrs,
        dtype=provider.dtype, comm=comm,
    ) as writer:
        writer.write_schedule(schedule.slice(lo, hi), lo, hi)
        writer.write_step(0, 0.0, solver.reach, solver.s, solver.status, lo, hi)
        itime = 1
        for k in range(n_steps):
            solver.step()
            if (k + 1) % every == 0 or k == n_steps - 1:
                writer.write_step(itime, solver.time, solver.reach, solver.s, solver.status, lo, hi)
                itime += 1
        writer.write_exits(solver.exit_time, solver.exit_reach, lo, hi)
    provider.close()
    if rank == 0 and not quiet:
        exited = int((solver.status == 2).sum())
        print(f"Done: {n_steps} steps, {itime} output times, {exited} of {solver.n} local particles exited", flush=True)
    return NetworkResults(out) if rank == 0 else None
```


- [ ] **Step 5: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_run.py -v` → 3 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/run.py src/fluvial_particle/network/results.py tests/network/test_run.py
git commit -m "Add run_network_simulation orchestration with startup diagnostics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Results: positions, arrivals, dataframes, summary

**Files:**
- Modify: `src/fluvial_particle/network/results.py`
- Test: `tests/network/test_results.py`

**Interfaces:**
- Produces on `NetworkResults`: `sources -> pd.DataFrame`, `start_time -> np.datetime64`, `network -> Network` (lazy, from the hydraulics file), `provider -> FileHydraulicsProvider` (lazy), `time_index(time) -> int`, `time_indices(time) -> NDArray[int64]`, `positions(time=None) -> pd.DataFrame | xr.Dataset`, `map_positions(time) -> pd.DataFrame`, `polylines() -> list`, `arrival_times(outlet=None) -> pd.DataFrame`, `arrival_histogram(outlet, bin_seconds) -> pd.DataFrame`, `to_dataframe(time=None) -> pd.DataFrame`, `summary() -> str`, `__repr__`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_results.py`:

```python
"""Tests for NetworkResults post-processing."""

import numpy as np
import pandas as pd
import pytest

from fluvial_particle.network.results import NetworkResults
from fluvial_particle.network.run import run_network_simulation
from tests.network.support import three_reach_dataset, write_network_file


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("res")
    path = write_network_file(tmp / "net.nc", three_reach_dataset())
    cfg = {
        "hydraulics_file": str(path), "dt": 600.0, "output_interval": 600.0, "end_time": "1979-01-01T02:00",
        "dispersion": {"model": "none"}, "mass_units": "g",
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 3.0, "particles": 3},
            {"reach_id": 102, "form": "slug", "time": 1800.0, "mass": 1.0, "particles": 1, "s_frac": 0.5},
        ],
    }
    res = run_network_simulation(cfg, tmp / "out", seed=1, quiet=True)
    yield res
    res.close()


def test_basics(run):
    assert run.n_particles == 4 and run.n_reach == 3
    assert run.start_time == np.datetime64("1979-01-01", "ns")
    assert isinstance(run.sources, pd.DataFrame) and len(run.sources) == 2
    assert run.time_index(np.datetime64("1979-01-01T00:10", "ns")) == 1
    assert run.time_index(-1) == 12
    assert list(run.time_indices([0, 2])) == [0, 2]
    assert list(run.time_indices(slice(0, 2))) == [0, 1]
    assert run.time_indices(None).size == 13
    assert "particles" in run.summary() and "NetworkResults" in repr(run)


def test_positions_and_map(run):
    df = run.positions(1)  # t = 600 s
    assert list(df.columns) == ["particle", "reach_index", "reach_id", "s", "status", "mass"]
    assert df.loc[0, "reach_id"] == 101 and df.loc[0, "s"] == 600.0 and df.loc[3, "status"] == 0
    mp = run.map_positions(1)
    # reach 0 polyline (-1000,500)->(0,0), hydraulic 1000 m: s=600 -> 60% along
    assert mp.loc[0, "x"] == pytest.approx(-400.0) and mp.loc[0, "y"] == pytest.approx(200.0)
    assert np.isnan(mp.loc[3, "x"])
    assert len(run.polylines()) == 3
    ds = run.positions()
    assert set(ds.data_vars) >= {"reach_index", "s", "status"}


def test_arrivals(run):
    df = run.arrival_times()
    assert len(df) == 4  # all exit within 2 h
    np.testing.assert_allclose(df.loc[df.particle < 3, "exit_time"], 2500.0)
    # reach 102: 1000 m left at 0.5 m/s = 2000 s, then 3000 m at 2 m/s = 1500 s, released at 1800
    assert df.loc[df.particle == 3, "exit_time"].item() == pytest.approx(1800.0 + 3500.0)
    assert set(df.columns) >= {"particle", "exit_time", "exit_datetime", "exit_reach", "exit_reach_id", "release_reach_id", "mass"}
    assert len(run.arrival_times(outlet=103)) == 4 and len(run.arrival_times(outlet=101)) == 0
    hist = run.arrival_histogram(103, bin_seconds=3600.0)
    assert list(hist.columns) == ["time_start", "time", "mass"]
    assert hist["mass"].sum() == pytest.approx(4.0)
    assert hist.loc[0, "mass"] == pytest.approx(3.0)


def test_to_dataframe(run):
    long = run.to_dataframe(1)
    assert len(long) == 4 and "time" in long.columns
    assert len(run.to_dataframe()) == 4 * 13
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_results.py -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Extend `results.py`**

Add imports: `import json`, `import pandas as pd`, `from .network import Network, NetworkBins`, `from .provider import FileHydraulicsProvider`. Add to `__init__`: `self._provider: FileHydraulicsProvider | None = None`, `self._network: Network | None = None`, `self._bins: dict[float, NetworkBins] = {}`. Extend `close()` to close `self._provider` when set. Then add:

```python
    def __repr__(self) -> str:
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
        t = np.datetime64(time, "ns")
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
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_results.py -v` → 4 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/results.py tests/network/test_results.py
git commit -m "Add NetworkResults positions, map positions, arrivals, and dataframes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Results: bins, counts, concentration, smoothing, persist

**Files:**
- Modify: `src/fluvial_particle/network/results.py`
- Test: `tests/network/test_results.py` (append)

**Interfaces:**
- Produces on `NetworkResults`: `bins(bin_length=100.0) -> NetworkBins`, `counts(time, bin_length=100.0) -> xr.DataArray`, `concentration(time, bin_length=100.0, smoothing=None) -> xr.DataArray` (`smoothing`: None, float meters, or `"auto"`), `reach_concentration(time, smoothing=None) -> xr.DataArray`, `persist(path, bin_length=100.0, smoothing=None) -> pathlib.Path`.

- [ ] **Step 1: Append failing tests**

```python
def test_counts_and_concentration(run):
    c = run.counts(1, bin_length=100.0)  # t = 600 s: three particles at s = 600 in reach 0 (width 10, depth 1)
    assert c.dims == ("bin",) and c.sum().item() == 3
    assert c.values[6] == 3
    conc = run.concentration(1, bin_length=100.0)
    assert conc.attrs["units"] == "g m-3"
    assert conc.values[6] == pytest.approx(3.0 / (10.0 * 1.0 * 100.0))
    assert conc.coords["reach_id"].values[6] == 101 and conc.coords["s_start"].values[6] == 600.0
    assert np.nansum(conc.values[:10]) == pytest.approx(conc.values[6])
    rc = run.reach_concentration(1)
    assert rc.sizes["bin"] == 3 and rc.values[0] == pytest.approx(3.0 / (10.0 * 1.0 * 1000.0))


def test_concentration_multi_time_and_smoothing(run):
    cube = run.concentration([0, 1, 2], bin_length=100.0)
    assert cube.dims == ("time", "bin") and cube.sizes["time"] == 3
    sm = run.concentration(1, bin_length=100.0, smoothing=150.0)
    bins = run.bins(100.0)
    vol = bins.bin_width * 10.0 * 1.0
    reach0 = bins.bin_reach == 0
    assert np.nansum(sm.values[reach0] * vol[reach0]) == pytest.approx(3.0)  # mass conserved in the reach
    assert (sm.values[reach0] > 0).sum() > 1  # spread over neighbors
    auto = run.concentration(1, bin_length=100.0, smoothing="auto")  # K = 0 -> tiny bandwidth -> same as binned
    assert np.nansum(auto.values[reach0] * vol[reach0]) == pytest.approx(3.0)


def test_dry_reach_is_nan(tmp_path):
    ds = three_reach_dataset(velocity=[1.0, 0.0, 2.0], flow_out=[10.0, 0.0, 80.0])
    path = write_network_file(tmp_path / "dry.nc", ds)
    cfg = {"hydraulics_file": str(path), "dt": 600.0, "output_interval": 600.0, "end_time": "1979-01-01T00:20",
           "dispersion": {"model": "none"},
           "sources": [{"reach_id": 102, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 1}]}
    with run_network_simulation(cfg, tmp_path / "out", seed=1, quiet=True) as res:
        conc = res.reach_concentration(1)
        assert np.isnan(conc.values[1]) and res.counts(1, np.inf).values[1] == 1


def test_persist(run, tmp_path):
    out = run.persist(tmp_path / "conc.nc", bin_length=500.0)
    import xarray as xr
    with xr.open_dataset(out, engine="h5netcdf") as ds:
        assert ds["concentration"].dims == ("time", "bin") and ds.sizes["time"] == 13
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_results.py -k "counts or smoothing or dry or persist" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Append to `results.py`**

Add import `from .dispersion import dispersion_coefficient`.

```python
    # ---- bins and concentration ---------------------------------------------
    def bins(self, bin_length: float = 100.0) -> NetworkBins:
        """Sub-reach bins of about ``bin_length`` meters (cached per length; np.inf gives one bin per reach)."""
        key = float(bin_length)
        if key not in self._bins:
            self._bins[key] = NetworkBins(self.network, key)
        return self._bins[key]

    def _bandwidth(self, i: int, bins: NetworkBins, smoothing: float | str | None) -> npt.NDArray[np.float64] | None:
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

    def _mass_per_bin(self, i: int, bins: NetworkBins, weights: npt.NDArray[np.float64], smoothing: float | str | None) -> npt.NDArray[np.float64]:
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
        units = f"{self._ds.attrs.get('mass_units', 'kg')} m-3"
        return self._wrap(np.stack(rows), idx, bins, time, "concentration", units)

    def reach_concentration(self, time: Any, smoothing: float | str | None = None) -> xr.DataArray:
        """concentration() with one bin per reach."""
        return self.concentration(time, bin_length=np.inf, smoothing=smoothing)

    def _wrap(self, data: npt.NDArray[Any], idx: npt.NDArray[np.int64], bins: NetworkBins, time: Any, name: str, units: str) -> xr.DataArray:
        coords = self._bin_coords(bins)
        single = not (time is None or isinstance(time, slice | list | tuple | np.ndarray))
        if single:
            return xr.DataArray(data[0], dims=("bin",), coords=coords, name=name, attrs={"units": units, "time": str(self.times[idx[0]])})
        coords["time"] = self.times[idx]
        return xr.DataArray(data, dims=("time", "bin"), coords=coords, name=name, attrs={"units": units})

    def persist(self, path: str | pathlib.Path, bin_length: float = 100.0, smoothing: float | str | None = None) -> pathlib.Path:
        """Write the full (time, bin) concentration cube to a NetCDF file and return its path."""
        da = self.concentration(None, bin_length=bin_length, smoothing=smoothing)
        da.attrs["bin_length"] = float(bin_length)
        da.to_dataset().to_netcdf(path, engine="h5netcdf")
        return pathlib.Path(path)
```

- [ ] **Step 4: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_results.py -v` → 8 passed.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/network/results.py tests/network/test_results.py
git commit -m "Add binned counts, concentration with optional smoothing, and persist

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: VTP/PVD export of network results

**Files:**
- Modify: `src/fluvial_particle/io/vtp_writer.py` (add `write_points`)
- Modify: `src/fluvial_particle/network/results.py` (add `to_vtp`)
- Test: `tests/test_io.py` (append one test), `tests/network/test_results.py` (append one test)

**Interfaces:**
- Produces: `VTPWriter.write_points(x, y, z, scalars: Mapping[str, np.ndarray], time: float, tidx: int, prefix: str = "particles") -> Path | None`; `NetworkResults.to_vtp(output_dir, times=None) -> pathlib.Path` (the `.pvd` path).

- [ ] **Step 1: Append failing tests**

To `tests/test_io.py` inside `class TestVTPWriter`:

```python
    def test_write_points_generic(self):
        with TemporaryDirectory() as tmpdir:
            writer = VTPWriter(pathlib.Path(tmpdir) / "vtp")
            x = np.array([0.0, 1.0, np.nan])
            scalars = {"reach_index": np.array([0, 1, -1], dtype=np.int64), "s": np.array([5.0, 6.0, np.nan])}
            vtp_file = writer.write_points(x, x, np.zeros(3), scalars, time=1.0, tidx=3, prefix="network")
            assert vtp_file.name == "network_0003.vtp"
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(str(vtp_file))
            reader.Update()
            pd_ = reader.GetOutput()
            assert pd_.GetNumberOfPoints() == 2
            assert pd_.GetPointData().GetArray("reach_index") is not None
```

To `tests/network/test_results.py`:

```python
def test_to_vtp(run, tmp_path):
    pvd = run.to_vtp(tmp_path / "vtk", times=[1, 2])
    assert pvd.name == "network.pvd" and pvd.exists()
    assert sorted(p.name for p in (tmp_path / "vtk" / "vtp").glob("*.vtp")) == ["network_0001.vtp", "network_0002.vtp"]
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/test_io.py::TestVTPWriter::test_write_points_generic tests/network/test_results.py::test_to_vtp -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add `write_points` to `VTPWriter`**

Insert after `write()`:

```python
    def write_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        scalars: Mapping[str, np.ndarray],
        time: float,
        tidx: int,
        prefix: str = "particles",
    ) -> Path | None:
        """Write arbitrary points with named scalar attributes to a VTP file.

        Args:
            x: point x coordinates; NaN marks points to skip.
            y: point y coordinates.
            z: point z coordinates.
            scalars: name -> per-point array (integer arrays are written as Int64, others as Float64).
            time: time value stored in field data.
            tidx: time step index used in the file name.
            prefix: file name prefix, ``<prefix>_<tidx:04d>.vtp``.

        Returns:
            Path to the written file, or None if no valid points.
        """
        valid_mask = ~np.isnan(x)
        n_valid = int(np.sum(valid_mask))
        if n_valid == 0:
            return None
        points = vtk.vtkPoints()
        coords = np.column_stack([x[valid_mask], y[valid_mask], z[valid_mask]]).astype(np.float64)
        points.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        verts = vtk.vtkCellArray()
        for i in range(n_valid):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        polydata.SetVerts(verts)
        for name, arr in scalars.items():
            data = np.asarray(arr)[valid_mask]
            dtype = np.int64 if data.dtype.kind in "iu" else None
            self._add_scalar(polydata, name, data, dtype=dtype)
        time_arr = vtk.vtkDoubleArray()
        time_arr.SetName("TimeValue")
        time_arr.SetNumberOfTuples(1)
        time_arr.SetValue(0, time)
        polydata.GetFieldData().AddArray(time_arr)
        vtp_file = self.output_dir / f"{prefix}_{tidx:04d}.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(vtp_file))
        writer.SetInputData(polydata)
        writer.SetDataModeToBinary()
        writer.Write()
        return vtp_file
```

Add `from collections.abc import Mapping` to the imports of `vtp_writer.py`.

- [ ] **Step 4: Add `to_vtp` to `NetworkResults`**

Add imports `from ..io import PVDWriter, VTPWriter`.

```python
    def to_vtp(self, output_dir: str | pathlib.Path, times: Any = None) -> pathlib.Path:
        """Write ``vtp/network_XXXX.vtp`` files and ``network.pvd`` for ParaView; returns the .pvd path."""
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
            f = vtp.write_points(df["x"].to_numpy(), df["y"].to_numpy(), np.zeros(self.n_particles), scalars,
                                 time=float(self.time_seconds[i]), tidx=int(i), prefix="network")
            if f is not None:
                pvd.add_timestep(float(self.time_seconds[i]), f)
        pvd.write()
        return out / "network.pvd"
```

- [ ] **Step 5: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/test_io.py tests/network/test_results.py -v` → all pass.

```bash
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/network
git add src/fluvial_particle/io/vtp_writer.py src/fluvial_particle/network/results.py tests/test_io.py tests/network/test_results.py
git commit -m "Add generic VTP point writer and NetworkResults.to_vtp

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: CLI entry points, package exports, template

**Files:**
- Modify: `src/fluvial_particle/cli.py`
- Modify: `src/fluvial_particle/network/__init__.py`
- Modify: `src/fluvial_particle/__init__.py`
- Test: `tests/network/test_cli.py`

**Interfaces:**
- Produces: `fluvial_particle.cli.network_serial()`, `network_mpi()`, `network_parser() -> argparse.ArgumentParser` (positional `settings_file`, `--output/-o` required, `--seed`, `--quiet`, `--init` writes the template to stdout and exits); package-level exports `run_network_simulation`, `NetworkConfig`, `NetworkResults`, `FileHydraulicsProvider`, `Network`, `NetworkBins`, `estimate_particles`, `get_network_config_template`.

- [ ] **Step 1: Write the failing tests**

`tests/network/test_cli.py`:

```python
"""Tests for the network CLI and package exports."""

import pytest

import fluvial_particle
from fluvial_particle.cli import network_parser, network_serial
from tests.network.support import three_reach_dataset, write_network_file


def test_exports():
    for name in ("run_network_simulation", "NetworkConfig", "NetworkResults", "FileHydraulicsProvider", "Network",
                 "NetworkBins", "estimate_particles", "get_network_config_template"):
        assert hasattr(fluvial_particle, name) and name in fluvial_particle.__all__


def test_parser_and_init(capsys):
    p = network_parser()
    ns = p.parse_args(["s.toml", "-o", "out", "--seed", "3", "--quiet"])
    assert ns.settings_file == "s.toml" and ns.output == "out" and ns.seed == 3 and ns.quiet
    with pytest.raises(SystemExit):
        network_serial(["--init"])
    assert "[network]" in capsys.readouterr().out


def test_network_serial_runs(tmp_path, monkeypatch):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    toml = tmp_path / "run.toml"
    toml.write_text("\n".join([
        "[network]", f'hydraulics_file = "{path}"', "dt = 600.0", "output_interval = 1200.0",
        'end_time = "1979-01-01T01:00"', "[network.dispersion]", 'model = "none"',
        "[[network.sources]]", "reach_id = 101", 'form = "slug"', "time = 0.0", "mass = 2.0", "particles = 2",
    ]) + "\n")
    network_serial([str(toml), "-o", str(tmp_path / "out"), "--seed", "1", "--quiet"])
    assert (tmp_path / "out" / "network_particles.nc").exists()
    with pytest.raises(FileNotFoundError):
        network_serial([str(tmp_path / "missing.toml"), "-o", str(tmp_path / "out")])
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n fluvial-particle pytest tests/network/test_cli.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Append to `cli.py`**

```python
import argparse
import pathlib
from collections.abc import Sequence


def network_parser() -> argparse.ArgumentParser:
    """Argument parser for the network solver entry points."""
    parser = argparse.ArgumentParser(
        prog="fluvial_particle_network",
        description="1D river-network particle tracking from a network hydraulics export.",
    )
    parser.add_argument("settings_file", nargs="?", help="TOML settings file with a [network] table")
    parser.add_argument("-o", "--output", help="output directory (created if missing)")
    parser.add_argument("--seed", type=int, default=None, help="base random seed")
    parser.add_argument("--quiet", action="store_true", help="suppress the startup report")
    parser.add_argument("--init", action="store_true", help="print a settings template and exit")
    return parser


def _network_args(argv: Sequence[str] | None) -> argparse.Namespace:
    from .network.config import get_network_config_template

    parser = network_parser()
    args = parser.parse_args(argv)
    if args.init:
        print(get_network_config_template())
        raise SystemExit(0)
    if args.settings_file is None or args.output is None:
        parser.error("settings_file and --output are required (unless using --init)")
    if not pathlib.Path(args.settings_file).exists():
        raise FileNotFoundError(f"Cannot find settings file {args.settings_file}")
    return args


def network_serial(argv: Sequence[str] | None = None) -> None:
    """Run the network solver in serial."""
    from .network.run import run_network_simulation

    args = _network_args(argv)
    run_network_simulation(args.settings_file, args.output, seed=args.seed, quiet=args.quiet)


def network_mpi(argv: Sequence[str] | None = None) -> None:
    """Run the network solver under MPI (mpiexec -n N fluvial_particle_network_mpi ...)."""
    from mpi4py import MPI

    from .network.run import run_network_simulation

    args = _network_args(argv)
    run_network_simulation(args.settings_file, args.output, seed=args.seed, comm=MPI.COMM_WORLD, quiet=args.quiet)
```

Put the three imports at the top of `cli.py` with the existing ones.

- [ ] **Step 4: Exports**

`src/fluvial_particle/network/__init__.py`:

```python
"""1D river-network particle tracking driven by network hydraulics exports."""

from .config import DispersionConfig, NetworkConfig, get_network_config_template
from .network import Network, NetworkBins
from .provider import FileHydraulicsProvider, HydraulicsProvider
from .results import NetworkResults
from .run import run_network_simulation
from .solver import NetworkSolver
from .sources import ParticleSchedule, estimate_particles, expand_sources


__all__ = [
    "DispersionConfig", "FileHydraulicsProvider", "HydraulicsProvider", "Network", "NetworkBins", "NetworkConfig",
    "NetworkResults", "NetworkSolver", "ParticleSchedule", "estimate_particles", "expand_sources",
    "get_network_config_template", "run_network_simulation",
]
```

In `src/fluvial_particle/__init__.py` add after the existing imports:

```python
from .network import (
    FileHydraulicsProvider,
    Network,
    NetworkBins,
    NetworkConfig,
    NetworkResults,
    estimate_particles,
    get_network_config_template,
    run_network_simulation,
)
```

and add those eight names to `__all__` (keep it sorted). Watch for a circular import: `network/run.py` imports `from .. import __version__`; since `__version__` is defined before the imports in the package `__init__`, that is fine, but if it fails, read the version from `importlib.metadata.version("fluvial-particle")` in `run.py` instead.

- [ ] **Step 5: Run, lint, mypy, commit**

Run: `conda run -n fluvial-particle pytest tests/network/test_cli.py tests/test_main.py -v` → all pass (the existing CLI tests still pass).

```bash
conda run -n fluvial-particle uv pip install -e ".[dev]"   # registers the console scripts
conda run -n fluvial-particle fluvial_particle_network --init | head -3
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle
git add src/fluvial_particle/cli.py src/fluvial_particle/__init__.py src/fluvial_particle/network/__init__.py tests/network/test_cli.py
git commit -m "Add network CLI entry points and package exports

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 18: Analytical acceptance tests

**Files:**
- Test: `tests/network/test_analytical.py`

**Interfaces:**
- Consumes: `run_network_simulation`, `chain_dataset`, `write_network_file`, scipy.stats.

- [ ] **Step 1: Write the tests**

```python
"""Analytical acceptance tests on a uniform chain: exact advection, Gaussian plume, inverse-Gaussian arrivals."""

import numpy as np
import pytest
from scipy import stats

from fluvial_particle.network.run import run_network_simulation
from tests.network.support import chain_dataset, write_network_file

pytestmark = pytest.mark.slow

N = 20000
K = 50.0
V = 1.0
L_REACH = 1000.0
N_REACH = 10
S0 = 500.0


def _run(tmp_path, ds, dt, end_seconds, output_interval, dispersion, release_s=0.0, particles=N):
    path = write_network_file(tmp_path / "chain.nc", ds)
    end = np.datetime64("1979-01-01", "ns") + np.timedelta64(int(end_seconds), "s")
    cfg = {
        "hydraulics_file": str(path), "dt": dt, "output_interval": output_interval, "end_time": str(end),
        "dispersion": dispersion,
        "sources": [{"reach_id": 1, "form": "slug", "time": 0.0, "mass": float(particles), "particles": particles, "s": release_s}],
    }
    return run_network_simulation(cfg, tmp_path / "out", seed=12345, quiet=True)


@pytest.mark.parametrize("dt", [900.0, 86400.0])
def test_pure_advection_exit_times_are_exact(tmp_path, dt):
    velocities = [1.0, 0.5, 2.0, 1.0, 4.0]
    ds = chain_dataset(n_reach=5, length=L_REACH, velocity=velocities)
    expected = sum(L_REACH / v for v in velocities)  # 4750 s
    with _run(tmp_path, ds, dt=dt, end_seconds=86400, output_interval=86400.0, dispersion={"model": "none"}, particles=100) as res:
        et = res.arrival_times()["exit_time"].to_numpy()
        assert et.size == 100
        np.testing.assert_allclose(et, expected, rtol=1e-9)


def test_gaussian_plume_moments_and_normality(tmp_path):
    ds = chain_dataset(n_reach=N_REACH, length=L_REACH, velocity=V, k_target=K)
    t_end = 3000.0
    with _run(tmp_path, ds, dt=10.0, end_seconds=t_end, output_interval=t_end, dispersion={"model": "constant", "value": K}, release_s=S0) as res:
        df = res.positions(-1)
        assert (df["status"] == 1).all()
        cum_before = np.arange(N_REACH) * L_REACH
        dist = cum_before[df["reach_index"].to_numpy()] + df["s"].to_numpy()
        mean, var = S0 + V * t_end, 2.0 * K * t_end
        assert abs(dist.mean() - mean) < 3.0 * np.sqrt(var / N)
        assert abs(dist.var() - var) < 3.0 * var * np.sqrt(2.0 / N)
        assert stats.normaltest(dist).pvalue > 0.01


def test_inverse_gaussian_arrival_times(tmp_path):
    ds = chain_dataset(n_reach=N_REACH, length=L_REACH, velocity=V, k_target=K)
    length = N_REACH * L_REACH - S0  # 9500 m to the outlet
    with _run(tmp_path, ds, dt=10.0, end_seconds=20000, output_interval=20000.0, dispersion={"model": "constant", "value": K}, release_s=S0) as res:
        et = res.arrival_times()["exit_time"].to_numpy()
        assert et.size == N, "every particle should have exited"
        mean = length / V
        shape = length**2 / (2.0 * K)
        dist = stats.invgauss(mean / shape, scale=shape)
        assert abs(et.mean() - mean) < 3.0 * np.sqrt(mean**3 / shape / N)
        assert stats.kstest(et, dist.cdf).pvalue > 0.01
```

- [ ] **Step 2: Run**

Run: `conda run -n fluvial-particle pytest tests/network/test_analytical.py -v`
Expected: 4 passed in well under a minute. If the KS test fails marginally, first check the exit-time bookkeeping in the solver (dispersive exits are stamped at the end of the step; with dt = 10 s that bias is a few thousandths in the CDF, below the 0.0115 critical value at N = 20000). Do not loosen the p-value.

- [ ] **Step 3: Lint and commit**

```bash
conda run -n fluvial-particle ruff check tests && conda run -n fluvial-particle ruff format tests
git add tests/network/test_analytical.py
git commit -m "Add analytical acceptance tests for advection and dispersion on a uniform chain

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 19: MPI path

**Files:**
- Create: `tests/network/mpi_run.py` (script run under mpiexec)
- Test: `tests/network/test_mpi.py`

**Interfaces:**
- Consumes: `run_network_simulation(..., comm=...)`, `NetworkWriter(comm=...)`.

- [ ] **Step 1: Write the script and the test**

`tests/network/mpi_run.py`:

```python
"""Run a network simulation under MPI: python mpi_run.py <hydraulics.nc> <output_dir>."""

import sys

from mpi4py import MPI

from fluvial_particle.network.run import run_network_simulation


def main() -> None:
    path, out = sys.argv[1], sys.argv[2]
    cfg = {
        "hydraulics_file": path, "dt": 600.0, "output_interval": 600.0, "end_time": "1979-01-01T02:00",
        "dispersion": {"model": "none"},
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 6.0, "particles": 6},
            {"reach_id": 102, "form": "loading", "rate": 0.001, "start": 0.0, "end": 3600.0, "particles": 5},
        ],
    }
    run_network_simulation(cfg, out, seed=1, comm=MPI.COMM_WORLD, quiet=True)


if __name__ == "__main__":
    main()
```

`tests/network/test_mpi.py`:

```python
"""Two-rank MPI write compared with the serial run (dispersion off, so results are identical)."""

import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from fluvial_particle.network.run import run_network_simulation
from tests.network.support import three_reach_dataset, write_network_file

mpi4py = pytest.importorskip("mpi4py")
h5py = pytest.importorskip("h5py")
if not h5py.get_config().mpi:
    pytest.skip("h5py without MPI support", allow_module_level=True)
if shutil.which("mpiexec") is None:
    pytest.skip("mpiexec not found", allow_module_level=True)


def test_two_ranks_match_serial(tmp_path):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    script = pathlib.Path(__file__).with_name("mpi_run.py")
    subprocess.run(["mpiexec", "-n", "2", sys.executable, str(script), str(path), str(tmp_path / "par")], check=True, timeout=300)
    cfg = {
        "hydraulics_file": str(path), "dt": 600.0, "output_interval": 600.0, "end_time": "1979-01-01T02:00",
        "dispersion": {"model": "none"},
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 6.0, "particles": 6},
            {"reach_id": 102, "form": "loading", "rate": 0.001, "start": 0.0, "end": 3600.0, "particles": 5},
        ],
    }
    run_network_simulation(cfg, tmp_path / "serial", seed=1, quiet=True).close()
    with xr.open_dataset(tmp_path / "par" / "network_particles.nc", engine="h5netcdf") as a, \
         xr.open_dataset(tmp_path / "serial" / "network_particles.nc", engine="h5netcdf") as b:
        for name in ("reach_index", "s", "status", "mass", "release_time", "exit_time", "exit_reach"):
            np.testing.assert_array_equal(a[name].values, b[name].values)
```

- [ ] **Step 2: Run**

Run: `conda run -n fluvial-particle pytest tests/network/test_mpi.py -v`
Expected: SKIPPED on the dev machine (h5py has no MPI). If an MPI-enabled environment is available (`conda create -n fp-mpi -c conda-forge "h5py=*=mpi_openmpi*" mpi4py ...`), the test must pass there; note the outcome in the commit message.

- [ ] **Step 3: Lint and commit**

```bash
conda run -n fluvial-particle ruff check tests && conda run -n fluvial-particle ruff format tests
git add tests/network/mpi_run.py tests/network/test_mpi.py
git commit -m "Add MPI two-rank test for the network writer (skipped without MPI h5py)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 20: Documentation, rules page, DRB smoke test

**Files:**
- Create: `docs/network.rst`
- Modify: `docs/index.rst` (add `network` after `output` in the toctree), `docs/optionsfile.rst` (append a `[network]` section), `docs/reference.rst` (append network API), `docs/output.rst` (append a pointer paragraph)
- Create: `.claude/rules/network.md`; Modify: `CLAUDE.md` (add the rule to the list and the subpackage to the structure)
- Create: `tests/network/test_drb.py`

- [ ] **Step 1: Write `docs/network.rst`**

```rst
1D River-Network Particle Tracking
==================================

The ``fluvial_particle.network`` subpackage tracks passive, non-reacting particles along a river network
using the *network hydraulics* NetCDF export produced by
``pywatershed.utils.export_network_hydraulics`` (or any file with the same schema). It is separate from
the 2D/3D VTK solver: a network run has its own configuration table, entry points, output file, and
results class.

Input
-----

One NetCDF file with dimensions ``reach``, ``time``, and optionally ``vertex``:

* static per reach: ``reach_id``, ``to_index`` (0-based downstream index, -1 at outlets), ``is_outlet``,
  ``length`` (m), ``slope``; optional polylines ``vertex_x``, ``vertex_y``, ``vertex_dist``,
  ``reach_vertex_start``, ``reach_vertex_count``;
* time-varying ``(time, reach)``: ``flow_in``, ``flow_out`` (m3/s), ``velocity`` (m/s), ``depth``,
  ``width`` (m), ``ustar`` (m/s), optional ``water_temperature``.

Where ``flow_out`` is 0 the hydraulics are 0 and particles wait. The provider validates variables,
units, and topology at open, streams two time slices at a time, and can subset reaches
(``reach_subset = [ids]`` or ``{outlet = id}``).

Particle convention
-------------------

A particle's state is ``(reach index, s)`` with ``0 <= s <= length`` from the reach's upstream end.
Each step: exact advection ``s += velocity * dt`` with hops to ``to_index`` carrying the unused time
(so any dt gives the same advective path), then one dispersive kick ``N(0, 1) * sqrt(2 K dt)`` with
displacement carry across boundaries. Upstream overshoot returns to the reach the particle came from,
or reflects at a headwater. At an outlet the particle exits and its exit time is recorded.

``K`` is the Fischer coefficient ``0.011 v^2 w^2 / (d u*)`` times ``dispersion.scale``, optionally
capped; ``model = "constant"`` and ``"none"`` are available.

Sources
-------

Every source is a mass loading at ``(reach_id, s)``:

* ``form = "slug"``: ``time``, ``mass``;
* ``form = "loading"``: constant ``rate`` on ``[start, end)``, or ``curve = [[time, rate], ...]``;
* ``form = "concentration"``: ``value`` on ``[start, end)`` or ``curve``; multiplied by the flow at the
  release point (interpolated between ``flow_in`` and ``flow_out``).

Particles sample the loading at equal-mass quantiles (or Poisson with ``spacing = "poisson"``). Set a
global ``particle_mass`` or a per-source ``particles`` count; ``estimate_particles`` reports the budget
for a target number of particles per bin.

Running
-------

.. code-block:: python

    from fluvial_particle import run_network_simulation, get_network_config_template

    results = run_network_simulation("network.toml", "./out", seed=42)
    print(results.summary())

or ``fluvial_particle_network network.toml -o out`` (``fluvial_particle_network_mpi`` under
``mpiexec``). ``--init`` prints the settings template.

Output and post-processing
--------------------------

``network_particles.nc`` holds ``reach_index``, ``s``, ``status`` on ``(time, particle)``, the release
table and exit table on ``particle``, and the run configuration as attributes. ``NetworkResults`` derives:

* ``positions(time)``, ``map_positions(time)``, ``polylines()``;
* ``counts(time, bin_length)``, ``concentration(time, bin_length, smoothing)`` on nearly uniform
  sub-reach bins (mass per ``width * depth * bin width``, in ``mass_units m-3``), ``persist()``;
* ``arrival_times(outlet)`` and ``arrival_histogram(outlet, bin_seconds)`` (breakthrough curves);
* ``to_dataframe()``, ``to_vtp()`` for ParaView.

MPI
---

Particles are split into contiguous slices across ranks; every rank holds the hydraulics. The writer
uses h5py's ``mpio`` driver, which requires an MPI-enabled h5py build.

Approximations
--------------

``K`` is sampled from the reach where the particle ends its advective move; a plain random walk across
a jump in ``K`` slightly over-populates the low-``K`` side; dispersive exits are stamped at the end of
the step. All are first order in ``dt``.
```

- [ ] **Step 2: Append a `[network]` section to `docs/optionsfile.rst`**

Add a heading ``Network solver settings`` at the end with a literal include of the template. Simplest: paste the output of `get_network_config_template()` in a ``.. code-block:: toml`` and a table of keys copied from the spec's configuration table (input, time, dispersion, sources, solver groups).

- [ ] **Step 3: Append to `docs/reference.rst`**

```rst
Network solver
--------------

.. automodule:: fluvial_particle.network.run
   :members:

.. automodule:: fluvial_particle.network.config
   :members:

.. automodule:: fluvial_particle.network.provider
   :members:

.. automodule:: fluvial_particle.network.network
   :members:

.. automodule:: fluvial_particle.network.sources
   :members:

.. automodule:: fluvial_particle.network.solver
   :members:

.. automodule:: fluvial_particle.network.results
   :members:
```

Add ``network`` to the toctree in `docs/index.rst` after ``output``, and append to `docs/output.rst`:

```rst
Network runs write ``network_particles.nc`` instead; see :doc:`network`.
```

- [ ] **Step 4: Rules page and CLAUDE.md**

`.claude/rules/network.md`:

```markdown
# 1D Network Solver (`fluvial_particle.network`)

Separate from the VTK 2D/3D path. Read the spec first:
`docs/superpowers/specs/2026-09-13-network-particle-solver-design.md`, and the upstream contract in the
pywatershed repo: `docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md`.

| Module | Purpose |
|---|---|
| `provider.py` | `FileHydraulicsProvider`: validate, subset, stream two time slices, hold/linear |
| `network.py` | `Network` topology and polylines; `NetworkBins` |
| `sources.py` | mass-loading sources -> `ParticleSchedule`; `estimate_particles` |
| `solver.py` | exact advection with time carry, one dispersive kick with displacement carry |
| `writer.py` | `network_particles.nc` via h5netcdf (mpio under MPI) |
| `results.py` | positions, bins, concentration, arrivals, VTP |
| `run.py` | `run_network_simulation`; CLI in `cli.py` |

Rules: dict keys and file variables use the export's names; solver never depends on bins or polylines;
tests use synthetic files from `tests/network/support.py`; the DRB file is not committed
(`FLUVIAL_PARTICLE_DRB_FILE` points at it for the optional test). Sample data:
`/home/rmcd/projects/pywatershed/examples/02a_network_hydraulics_export/drb_network_hydraulics.nc`.
```

In `CLAUDE.md`: add `├── network/          # 1D river-network solver (see .claude/rules/network.md)` to the structure block and `- network.md - 1D network solver modules and data locations` to the rules list.

- [ ] **Step 5: DRB smoke test**

`tests/network/test_drb.py`:

```python
"""End-to-end run on the DRB export when FLUVIAL_PARTICLE_DRB_FILE is set."""

import os
import pathlib

import numpy as np
import pytest

from fluvial_particle.network import FileHydraulicsProvider, Network, run_network_simulation

DRB = os.environ.get("FLUVIAL_PARTICLE_DRB_FILE")
pytestmark = pytest.mark.skipif(not DRB or not pathlib.Path(DRB).exists(), reason="DRB file not available")


def test_drb_five_days_from_headwaters(tmp_path):
    with FileHydraulicsProvider(DRB) as prov:
        net = Network(prov.static)
        heads = net.headwaters()
        start = prov.times[0]
    sources = [{"reach_id": int(r), "form": "slug", "time": 0.0, "mass": 10.0} for r in heads]
    sources.append({"reach_id": 4205, "form": "slug", "time": 0.0, "mass": 10.0})  # guarantees an exit at Trenton
    cfg = {
        "hydraulics_file": DRB, "dt": 900.0, "output_interval": 3600.0,
        "start_time": str(start), "end_time": str(start + np.timedelta64(5, "D")),
        "particle_mass": 1.0,
        "sources": sources,
    }
    with run_network_simulation(cfg, tmp_path / "drb", seed=1, quiet=True) as res:
        assert res.n_particles == 10 * (heads.size + 1)
        status = res.positions(-1)["status"]
        assert status.isin([1, 2]).all()
        assert len(res.arrival_times(outlet=4205)) > 0
        conc = res.reach_concentration(-1)
        assert np.nanmax(conc.values) > 0
```

- [ ] **Step 6: Build docs, run everything, commit**

```bash
conda run -n fluvial-particle sphinx-build -b html docs docs/_build/html 2>&1 | tail -3
FLUVIAL_PARTICLE_DRB_FILE=/home/rmcd/projects/pywatershed/examples/02a_network_hydraulics_export/drb_network_hydraulics.nc conda run -n fluvial-particle pytest tests -q
conda run -n fluvial-particle pre-commit run --all-files
git add docs .claude/rules/network.md CLAUDE.md tests/network/test_drb.py
git commit -m "Document the network solver and add the DRB smoke test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 21: DRB demo notebook

**Files:**
- Create: `notebooks/network-drb-demo.ipynb`

**Interfaces:**
- Consumes: the public API from Task 17 and the DRB file.

- [ ] **Step 1: Generate the notebook with nbformat**

Run this script from the repo root (`conda run -n fluvial-particle python build_demo.py`, then delete the script):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# 1D network particle tracking on the Delaware River Basin

Passive tracer demo driven by pywatershed's network hydraulics export. The input file is produced by
`examples/02a_network_hydraulics_export.ipynb` in the pywatershed repository; set its path below.""")
code("""import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from fluvial_particle import (FileHydraulicsProvider, Network, NetworkConfig, estimate_particles,
                              run_network_simulation)

DRB = pathlib.Path("/home/rmcd/projects/pywatershed/examples/02a_network_hydraulics_export/drb_network_hydraulics.nc")
OUT = pathlib.Path("./network-drb-demo-output")
TRENTON = 4205""")
md("## 1. The network")
code("""prov = FileHydraulicsProvider(DRB)
net = Network(prov.static, crs_wkt=prov.crs_wkt)
print(f"{net.n_reach} reaches, {net.outlets().size} outlets, {net.headwaters().size} headwaters, "
      f"{net.length.sum()/1000:.0f} km; {prov.times[0]} .. {prov.times[-1]}")
last = prov.hydraulics(prov.times[-1])
fig, ax = plt.subplots(figsize=(7, 9))
lc = LineCollection([np.column_stack(p) for p in net.polylines()], array=last["velocity"], cmap="viridis", linewidths=1.2)
ax.add_collection(lc); ax.autoscale(); ax.set_aspect("equal"); fig.colorbar(lc, label="velocity (m/s), last day")
ax.set_title("DRB network"); plt.show()""")
md("## 2. Sources: a slug at every headwater plus a week-long loading on the mainstem")
code("""start = np.datetime64("1979-03-01")
heads = net.headwaters()
mainstem = [r for r in net.upstream_of(TRENTON)][20]  # a mainstem reach well upstream of Trenton
sources = [{"reach_id": int(r), "form": "slug", "time": 0.0, "mass": 100.0} for r in heads]
sources.append({"reach_id": int(mainstem), "form": "loading", "rate": 0.05, "start": "1979-03-03", "end": "1979-03-10"})
cfg = NetworkConfig.from_dict({
    "hydraulics_file": str(DRB), "start_time": str(start), "end_time": str(start + np.timedelta64(30, "D")),
    "dt": 900.0, "output_interval": 3600.0, "particle_mass": 1.0, "mass_units": "kg",
    "dispersion": {"model": "fischer"}, "sources": sources, "seed": 42,
})
budget = estimate_particles(cfg, prov, target_per_bin=50, bin_length=500.0)
print(budget.tail(3)); print("total particles at 1 kg/particle:", int(budget.particles.sum()))""")
md("## 3. Run one month")
code("""res = run_network_simulation(cfg, OUT)
print(res.summary())""")
md("## 4. Arrival-time distributions at the outlets")
code("""fig, ax = plt.subplots(figsize=(8, 4))
for outlet in net.outlets():
    h = res.arrival_histogram(int(outlet), bin_seconds=6 * 3600.0)
    if h.mass.sum() > 0:
        ax.plot(h.time, h.mass, label=f"reach {outlet}" + (" (Trenton)" if outlet == TRENTON else ""))
ax.set_ylabel("mass arriving per 6 h (kg)"); ax.legend(); ax.set_title("Breakthrough at the outlets")
released = res.positions(-1).mass.sum(); recovered = res.arrival_times().mass.sum()
print(f"released {released:.0f} kg, recovered at outlets {recovered:.0f} kg ({100*recovered/released:.0f}%)")""")
md("## 5. Concentration along the mainstem")
code("""main_ids = net.upstream_of(TRENTON)
# walk the mainstem from Trenton upstream following the largest-flow parent
path_idx = [net.index_of(TRENTON)]
while True:
    parents = net.parents(path_idx[-1])
    if parents.size == 0:
        break
    path_idx.append(int(parents[np.argmax(last["flow_out"][parents])]))
path_idx = path_idx[::-1]
bins = res.bins(500.0)
sel = np.isin(bins.bin_reach, path_idx)
order = np.argsort([path_idx.index(r) for r in bins.bin_reach[sel]], kind="stable")
cube = res.concentration(None, bin_length=500.0, smoothing="auto").values[:, sel][:, order]
dist = np.cumsum(bins.bin_width[sel][order]) / 1000.0
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.pcolormesh(dist, res.times, cube, shading="nearest", cmap="magma")
fig.colorbar(im, label="concentration (kg/m3)"); ax.set_xlabel("distance along mainstem (km)"); plt.show()""")
md("## 6. Map animation")
code("""fig, ax = plt.subplots(figsize=(7, 9))
ax.add_collection(LineCollection([np.column_stack(p) for p in net.polylines()], colors="lightgray", linewidths=0.8))
ax.autoscale(); ax.set_aspect("equal")
scat = ax.scatter([], [], s=4, c="crimson")
title = ax.set_title("")
frames = range(0, res.times.size, 3)

def update(i):
    df = res.map_positions(int(i))
    ok = df.status == 1
    scat.set_offsets(np.column_stack([df.x[ok], df.y[ok]]))
    title.set_text(str(res.times[i])[:16])
    return scat, title

anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
anim.save(OUT / "drb_particles.gif", writer="pillow", fps=12)
plt.close(fig)
from IPython.display import Image
Image(filename=str(OUT / "drb_particles.gif"))""")
code("res.close(); prov.close()")
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/network-drb-demo.ipynb")
```

If `nbformat` is missing: `conda run -n fluvial-particle uv pip install nbformat`. If `pillow` is missing for the GIF writer, install it or switch to `anim.to_jshtml()`.

- [ ] **Step 2: Execute the notebook once**

Run: `conda run -n fluvial-particle jupyter nbconvert --to notebook --execute --inplace notebooks/network-drb-demo.ipynb --ExecutePreprocessor.timeout=1800` (install `jupyter` into the env with uv if absent). Confirm every cell ran, the GIF exists, and the recovered-mass line is sensible (most slug mass reaches Trenton within the month; the loading source is partly still in transit).

- [ ] **Step 3: Strip outputs and commit**

The repo's pre-commit runs `nbstripout`, so outputs are removed at commit time; keep the GIF out of git (add `notebooks/network-drb-demo-output/` to `.gitignore`).

```bash
echo "notebooks/network-drb-demo-output/" >> .gitignore
conda run -n fluvial-particle pre-commit run --all-files
git add notebooks/network-drb-demo.ipynb .gitignore
git commit -m "Add DRB network particle demo notebook

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Not in this plan

- Version bump to 0.1.0 (`uv run bump-my-version bump minor`) happens when the branch merges, per the spec.
- The production in-memory provider and BMI facade, the NWM transformer, drift-corrected random walks, and distributaries are extension points listed in the spec.

## Self-review notes

- **Spec coverage:** provider (Tasks 5-6), network and bins (3-4), sources and budget (8-9), solver step (10-11), dispersion (2), writer and MPI (12, 19), results and post-processing (14-16), config and CLI (7, 17), diagnostics (13), analytical tests (18), docs, rules, DRB test (20), demo (21), packaging (1). The spec's `conventions_note` attribute copy is in Task 13 via the provider attribute added there.
- **Type consistency:** `hydraulics(t)` returns `dict[str, ndarray]` everywhere; `ParticleSchedule.slice(lo, hi)` and `.n` are used by Tasks 10, 12, 13; `NetworkBins.bin_of`, `.bin_reach`, `.bin_width`, `.s_start`, `.s_end`, `.reach_bin_start`, `.bins_per_reach` are used in Task 15 exactly as defined in Task 4; `DispersionConfig` field names (`model`, `scale`, `cap`, `value`) match `dispersion_coefficient` keyword names; `NetworkWriter.write_step(itime, time_seconds, reach, s, status, lo, hi)` is called identically in Tasks 12, 13, 19.
- **Sources:** `_concentration_rate` (Task 8) builds the C(t)·Q(t) product on the union of curve breakpoints and provider timestamps; both `expand_sources` and `estimate_particles` call it, so the two agree on total mass.
