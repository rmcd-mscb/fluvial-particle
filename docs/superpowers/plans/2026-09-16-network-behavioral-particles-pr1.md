# Network Behavioral Particles, PR 1: Seams, Vertical Model, Drift Particles

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the network solver extensible the way the 2D/3D solver is (registry-named particle models with hooks and declared state), and ship the first model: quasi-2D drift particles with a configurable vertical velocity and mixing profile, sub-stepped vertical random walk with the gradient drift term, Robin-boundary deposition, and a shear-dispersion correction computed from the chosen profiles.

**Architecture:** `NetworkSolver` stays the base class and the passive model. Two hooks (`on_release`, `behave`) and a per-particle velocity factor in `_advect` are the only changes to the transport kernel. Models declare per-particle state as `StateVar` tuples; the base allocates it, the writer and results handle it generically. `PARTICLE_MODELS` maps config names to classes; `[network.particles]` names the model and holds its parameters. `[network.dispersion]` gains a background term, a shear-correction switch, and a `vertical` sub-table mirroring the 2D/3D `lev + beta u* h` parameterization. `vertical.py` holds pure functions for the profiles, the walk, the sub-step count, Taylor's shear-dispersion integral, and the deposition probability. `random_walk.reflect_interval` is extracted from `Particles.validate_z` and shared by both solvers.

**Tech Stack:** as the base plan (Python 3.10+, numpy, scipy for the quadrature, xarray, h5netcdf, pytest, ruff, mypy).

**Spec:** `docs/superpowers/specs/2026-09-16-network-behavioral-particles-design.md`, Decisions 1 to 8 including 7a. **Base spec:** `docs/superpowers/specs/2026-09-13-network-particle-solver-design.md`. **Base plan:** `docs/superpowers/plans/2026-09-13-network-particle-solver.md` (conventions, file structure, test support).

## Global Constraints

- Everything in the base plan's Global Constraints still holds: conda env, mypy strict on `src/`, ruff google docstrings at 120 columns, export names for dict keys and file variables, commit after every task with the attribution line.
- **The passive model is bit-identical to today.** With no `[network.particles]` table, every existing test in `tests/network/` passes unchanged, and the RNG draw sequence of `NetworkSolver.step` is unchanged (the drift model draws its normals inside `behave`, before `_disperse`; the base `behave` draws nothing).
- **Models never touch the writer or results.** They declare state; the writer creates variables from the declaration; results read them by an attribute marker.
- **The solver never depends on bins or polylines** (base spec rule); nothing in this PR imports `NetworkBins` into `solver.py` or `particles.py`.
- Vertical convention: `zeta = z / h`, 0 at the bed, 1 at the surface, walk domain `[zeta_min, 1 - zeta_min]`, default `zeta_min = 0.001`. Vertical velocity `w` is positive downward.
- Every quadrature and every walk uses the **same** floored, renormalized velocity factor and the **same** truncated domain, so the shear correction removes exactly what the walk generates.
- Status codes: 0 unreleased, 1 active, 2 exited, 3 settled, 4 removed; every code above 1 is terminal.
- Branch: `feature/network-behavioral-particles` cut from `spec/network-behavioral-particles` (PR #53, which holds the spec and this plan), rebased onto `main` once #53 merges.
- New runtime dependency: none (`scipy` is already a dependency for the 2D/3D path; confirm with `grep scipy pyproject.toml` in Task 1 and add it to `dependencies` if it is only in an extra).

---

## File Structure

| Path | Change | Responsibility |
|---|---|---|
| `src/fluvial_particle/random_walk.py` | create | `reflect_interval(x, lo, hi)`: fold positions back into `[lo, hi]`, any number of crossings |
| `src/fluvial_particle/Particles.py` | modify | `validate_z` calls `reflect_interval`; behavior unchanged |
| `src/fluvial_particle/network/solver.py` | modify | `Status` gains `SETTLED`, `REMOVED`; hooks `on_release`, `behave`; velocity factor in `_advect`; declared-state allocation and `state` views; shear correction of `K`; `validate_params` classmethod; `resolves_vertical` flag |
| `src/fluvial_particle/network/particles.py` | create | `StateVar`; `PARTICLE_MODELS`; `resolve_model`; `DriftParticles` |
| `src/fluvial_particle/network/vertical.py` | create | `VerticalProfiles` (velocity factor, `Kz`, `dKz/dz`, `Kz_max`, normalization and shear-coefficient tables), `substep_count`, `deposition_probability`, `shear_dispersion_coefficient` |
| `src/fluvial_particle/network/dispersion.py` | modify | `background` term in `dispersion_coefficient` |
| `src/fluvial_particle/network/config.py` | modify | `VerticalDispersionConfig`; `DispersionConfig` gains `background`, `shear_correction`, `vertical`; `ParticlesConfig`; `NetworkConfig.particles`; template |
| `src/fluvial_particle/network/writer.py` | modify | declared-state variables; `status` `flag_meanings`; `particle_model` attrs; `write_step(..., state=)` |
| `src/fluvial_particle/network/results.py` | modify | `state_variables`, `positions` with state, `terminal(status)`, `profile()`; `arrival_times` filters on `EXITED` |
| `src/fluvial_particle/network/run.py` | modify | resolve the model class; pass params; write state; diagnostics for sub-steps and shear correction |
| `src/fluvial_particle/network/__init__.py` | modify | export `StateVar`, `PARTICLE_MODELS`, `DriftParticles`, `ParticlesConfig`, `VerticalDispersionConfig` |
| `tests/test_random_walk.py` | create | fold tests |
| `tests/network/test_solver.py` | modify | hooks, factor, terminal statuses, state allocation |
| `tests/network/test_particles.py` | create | `StateVar`, registry, `DriftParticles` unit behavior |
| `tests/network/test_vertical.py` | create | profiles, quadrature, sub-steps, deposition probability |
| `tests/network/test_config.py`, `test_writer.py`, `test_results.py`, `test_run.py` | modify | new tables, state round trip, `terminal`, `profile`, model wiring |
| `tests/network/test_analytical.py` | modify | acceptance tests 1 to 4 and 7 of the spec |
| `docs/network.rst`, `HISTORY.md`, `.claude/rules/network.md` | modify | "Behavioral particles" section; changelog; rules |

---

### Task 1: Branch, shared reflection fold

**Files:**
- Create: `src/fluvial_particle/random_walk.py`
- Modify: `src/fluvial_particle/Particles.py` (`validate_z`)
- Test: `tests/test_random_walk.py`

**Interfaces:**
- Produces: `reflect_interval(x: NDArray[float64], lo: ArrayLike, hi: ArrayLike) -> NDArray[float64]`: returns the fold of `x` into `[lo, hi]` elementwise (`lo`, `hi` broadcast); positions already inside are returned unchanged; NaN passes through; where `hi <= lo` the result is `lo`.

- [ ] **Step 1: Branch**

```bash
git checkout spec/network-behavioral-particles && git pull
git checkout -b feature/network-behavioral-particles
grep -n scipy pyproject.toml   # must be in [project].dependencies; add it if only in an extra
```

- [ ] **Step 2: Failing tests** in `tests/test_random_walk.py`

```python
"""Tests for the shared interval reflection used by both solvers."""

import numpy as np

from fluvial_particle.random_walk import reflect_interval


def test_inside_unchanged_and_single_crossing():
    x = np.array([0.5, -0.2, 1.3])
    np.testing.assert_allclose(reflect_interval(x, 0.0, 1.0), [0.5, 0.2, 0.7])


def test_multiple_crossings_fold():
    # 2.3 crosses the unit interval twice: 2.3 -> 1.7 (about hi) -> 0.3 (about lo)
    np.testing.assert_allclose(reflect_interval(np.array([2.3, -3.1]), 0.0, 1.0), [0.3, 0.9])


def test_broadcast_bounds_nan_and_degenerate():
    x = np.array([1.5, np.nan, 0.7])
    lo = np.array([1.0, 0.0, 0.5])
    hi = np.array([2.0, 1.0, 0.5])
    out = reflect_interval(x, lo, hi)
    assert out[0] == 1.5 and np.isnan(out[1]) and out[2] == 0.5


def test_matches_particles_validate_z_fold():
    # Same arithmetic as the fold that was inline in Particles.validate_z.
    rng = np.random.RandomState(0)
    x = rng.uniform(-3, 4, 1000)
    span = 1.0
    u = np.mod(x - 0.0, 2 * span)
    expected = np.where(u > span, 2 * span - u, u)
    np.testing.assert_allclose(reflect_interval(x, 0.0, 1.0), expected)
```

- [ ] **Step 3: Run to verify failure**: `conda run -n fluvial-particle pytest tests/test_random_walk.py -q` → `ModuleNotFoundError`.

- [ ] **Step 4: Implement** `src/fluvial_particle/random_walk.py`

```python
"""Numerical pieces of the random walk shared by the 2D/3D and network solvers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def reflect_interval(
    x: npt.NDArray[np.float64], lo: npt.ArrayLike, hi: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Fold positions back into ``[lo, hi]`` by mirror reflection, handling any number of crossings.

    Mirror reflection is the no-flux wall of a passive random walk; clamping instead piles particles
    onto the bounds (see Particles.validate_z). NaN positions pass through; where ``hi <= lo`` the
    interval has no width and the result is ``lo``.

    Args:
        x: positions.
        lo: lower bound(s), broadcast against ``x``.
        hi: upper bound(s), broadcast against ``x``.

    Returns:
        The reflected positions, a new array.
    """
    x = np.asarray(x, dtype=np.float64)
    lo_a = np.broadcast_to(np.asarray(lo, dtype=np.float64), x.shape)
    hi_a = np.broadcast_to(np.asarray(hi, dtype=np.float64), x.shape)
    span = hi_a - lo_a
    out = x.copy()
    finite = np.isfinite(x)
    a = finite & (span > 0.0) & ((x < lo_a) | (x > hi_a))
    if a.any():
        u = np.mod(x[a] - lo_a[a], 2.0 * span[a])
        out[a] = lo_a[a] + np.where(u > span[a], 2.0 * span[a] - u, u)
    b = finite & np.isfinite(span) & (span <= 0.0)
    out[b] = lo_a[b]
    return out
```

Then in `Particles.validate_z`, replace the two inline blocks (the `a` fold and the `b` pin) with:

```python
        lo = self.bedelev + self.vertbound * self.depth
        hi = self.wse - self.vertbound * self.depth
        pz[:] = reflect_interval(pz, lo, hi)
```

keeping the explanatory comment about clamping versus reflecting, and adding `from .random_walk import reflect_interval` at the top of `Particles.py`. Deactivated particles carry NaN and pass through, as before.

- [ ] **Step 5: Verify, lint, commit**

```bash
conda run -n fluvial-particle pytest tests/test_random_walk.py tests/test_analytical.py tests/test_particles.py -q
conda run -n fluvial-particle ruff check src tests && conda run -n fluvial-particle ruff format src tests
conda run -n fluvial-particle mypy src/fluvial_particle/random_walk.py
git add -A && git commit -m "Extract the interval reflection fold to random_walk.py; Particles.validate_z uses it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

The 2D/3D vertical analytical tests (`tests/test_analytical.py`) must pass unchanged; they are the regression guard for the extraction.

---

### Task 2: Terminal statuses

**Files:**
- Modify: `src/fluvial_particle/network/solver.py` (`Status`), `writer.py` (`flag_meanings`), `results.py` (`terminal`, `arrival_times`)
- Test: `tests/network/test_solver.py`, `tests/network/test_results.py`

**Interfaces:**
- `Status.SETTLED = 3`, `Status.REMOVED = 4`, module aliases `SETTLED`, `REMOVED`, and `TERMINAL_STATUSES = (EXITED, SETTLED, REMOVED)`.
- `NetworkSolver.terminate(idx, status, t_end)`: sets `status`, `exit_time = t_end`, `exit_reach = reach[idx]`; keeps `s` (settled particles have a bed position) and `reach`. Used by models; the base uses it nowhere (exits keep their existing inline bookkeeping so the passive RNG and arithmetic are untouched).
- `NetworkResults.terminal(status: int) -> pd.DataFrame`: particles whose final status equals `status`, with `exit_time`, `exit_reach`, `reach_id`, `release_*`, `mass`; `arrival_times` becomes `terminal(EXITED)` plus the outlet filter.

- [ ] **Step 1: Failing tests**

`tests/network/test_solver.py` (append):

```python
def test_terminate_marks_settled_and_keeps_position(three_reach_solver):
    solver = three_reach_solver(n=3)
    solver.step()
    solver.terminate(np.array([1]), SETTLED, solver.time)
    assert solver.status[1] == 3
    assert solver.exit_time[1] == solver.time
    assert solver.exit_reach[1] == solver.reach[1]
    assert np.isfinite(solver.s[1])
    solver.step()
    # a terminal particle no longer moves
    assert solver.status[1] == 3


def test_conservation_over_all_statuses(three_reach_solver):
    solver = three_reach_solver(n=50)
    for _ in range(20):
        solver.step()
        counts = np.bincount(solver.status, minlength=5)
        assert counts.sum() == solver.n
```

(`three_reach_solver` is whatever fixture `test_solver.py` already uses to build a solver on the three-reach dataset; reuse it, do not add a second one.)

`tests/network/test_results.py` (append): write a small file with the writer where one particle has status 3 at the last time and `exit_*` set, then `terminal(3)` returns exactly that particle, `arrival_times()` does not include it, and `status` has `flag_meanings`.

- [ ] **Step 2: Run to verify failure**, then implement:

`solver.py`:

```python
class Status(IntEnum):
    UNRELEASED = 0
    ACTIVE = 1
    EXITED = 2      # left the network at an outlet
    SETTLED = 3     # deposited on the bed (behavioral models)
    REMOVED = 4     # taken out by a model (mass floor, mortality)

TERMINAL_STATUSES = (Status.EXITED, Status.SETTLED, Status.REMOVED)
```

```python
    def terminate(self, idx: IntArray, status: int, t_end: float) -> None:
        """Give particles ``idx`` a terminal status at time ``t_end`` (they keep reach and s)."""
        if idx.size == 0:
            return
        self._status[idx] = status
        self._exit_time[idx] = t_end
        self._exit_reach[idx] = self._reach[idx]
```

`writer.py`: on the `status` variable set `v.attrs["flag_values"] = np.array([0, 1, 2, 3, 4], dtype=np.int8)` and `v.attrs["flag_meanings"] = "unreleased active exited settled removed"`; change `exit_time`/`exit_reach` long names to "time the particle reached a terminal status" / "reach at the terminal status: the outlet for exited, the bed reach for settled".

`results.py`: factor the body of `arrival_times` into `terminal(status)`; `arrival_times(outlet=None)` calls `terminal(2)` and applies the outlet filter. Final status is `self._ds["status"].values[-1]`.

- [ ] **Step 3: Verify, lint, commit**: all of `tests/network` passes.

```bash
git commit -am "Add SETTLED and REMOVED terminal statuses; NetworkResults.terminal

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Solver hooks and the velocity factor

**Files:**
- Modify: `src/fluvial_particle/network/solver.py`
- Test: `tests/network/test_solver.py`

**Interfaces:**
- `NetworkSolver.on_release(idx: IntArray, h: Mapping[str, FloatArray]) -> None`: no-op in the base.
- `NetworkSolver.behave(h, tau: FloatArray, t: float, dt: float) -> FloatArray | None`: no-op returning `None` in the base.
- `_release(t, dt, h)` calls `on_release(new_idx, h)` after activating.
- `_advect(v, tau, t, dt, factor: FloatArray | None)`: per-particle velocity `u = v[reach] * factor` (factor 1 when `None`); hop time carry uses `u` on both sides.
- `step()` order: `h` → `tau = _release` → `factor = behave` → `k` → `_advect(..., factor)` → `_disperse`.

- [ ] **Step 1: Failing tests** (append to `test_solver.py`)

```python
class HalfSpeed(NetworkSolver):
    """Test model: records releases and advects at half the reach velocity."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.released = []

    def on_release(self, idx, h):
        self.released.append(idx.copy())

    def behave(self, h, tau, t, dt):
        return np.full(self.n, 0.5)


def test_hooks_called_and_factor_halves_the_landing(chain_provider_factory):
    # Two reaches of 100 m at 1 m/s, dt = 150 s, one particle released at s = 0 of reach 0.
    # Base: 150 m -> reach 1 at s = 50. Half speed: 75 m -> reach 0 at s = 75.
    base = make_solver(NetworkSolver, ...)
    half = make_solver(HalfSpeed, ...)
    base.step(); half.step()
    assert base.reach[0] == 1 and base.s[0] == pytest.approx(50.0)
    assert half.reach[0] == 0 and half.s[0] == pytest.approx(75.0)
    assert len(half.released) == 1 and list(half.released[0]) == [0]


def test_factor_applies_across_a_hop(...):
    # dt = 300 s, half speed: 150 m -> crosses at 200 s of budget, lands in reach 1 at s = 25.
    ...
    assert half.reach[0] == 1 and half.s[0] == pytest.approx(25.0)


def test_behave_returning_none_equals_ones(...):
    class Ones(NetworkSolver):
        def behave(self, h, tau, t, dt):
            return np.ones(self.n)
    # same seed, 30 steps with Fischer dispersion on the three-reach set: identical arrays
    ...
```

Use the existing `ArrayHydraulicsProvider` and `chain_dataset` from `tests/network/support.py` to build the two-reach case with `model = "none"`.

- [ ] **Step 2: Run to verify failure**, then implement in `solver.py`:

```python
    def step(self) -> None:
        t = self.time
        dt = self.dt
        h = self.provider.hydraulics(self.midpoint_time())
        tau = self._release(t, dt, h)
        factor = self.behave(h, tau, t, dt)
        v = np.asarray(h["velocity"], dtype=np.float64)
        d = self.dispersion
        k = dispersion_coefficient(h, d.model, scale=d.scale, cap=d.cap, value=d.value, background=d.background)
        k = self._correct_shear(k, h)          # Task 6; identity until then
        self._advect(v, tau, t, dt, factor)
        self._disperse(k, tau, t, dt, np.asarray(h["flow_out"], dtype=np.float64))
        self.time = t + dt

    def on_release(self, idx: IntArray, h: Mapping[str, FloatArray]) -> None:
        """Hook: initialize model state for particles released this step (no-op in the base)."""

    def behave(self, h: Mapping[str, FloatArray], tau: FloatArray, t: float, dt: float) -> FloatArray | None:
        """Hook: update model state before advection; return a per-particle velocity factor or None."""
        return None
```

`_advect` with the factor:

```python
    def _advect(self, v, tau, t, dt, factor=None):
        idx = np.nonzero(self._status == ACTIVE)[0]
        if idx.size == 0:
            return
        f = np.ones(self.n) if factor is None else np.asarray(factor, dtype=np.float64)
        self._s[idx] += f[idx] * v[self._reach[idx]] * tau[idx]
        for _ in range(self.max_hops):
            over = self._s[idx] > self._length[self._reach[idx]]
            if not over.any():
                return
            j = idx[over]
            rj = self._reach[j].astype(np.int64)
            uj = f[j] * v[rj]                      # > 0: s only grew past length because uj > 0
            time_left = (self._s[j] - self._length[rj]) / uj
            ...  # exits as before
            m = j[~exiting]
            self._reach[m] = nxt[~exiting]
            self._s[m] = f[m] * v[nxt[~exiting]] * time_left[~exiting]
            idx = m
```

When `factor is None` the arithmetic is `1.0 * v`, bit-identical to today.

- [ ] **Step 3: Verify, lint, commit**: whole `tests/network` passes, including the analytical tests.

```bash
git commit -am "Add on_release and behave hooks and a per-particle velocity factor to NetworkSolver

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Declared state through solver, writer, and results

**Files:**
- Create: `src/fluvial_particle/network/particles.py` (`StateVar` only for now)
- Modify: `solver.py`, `writer.py`, `results.py`, `run.py`
- Test: `tests/network/test_particles.py`, `test_writer.py`, `test_results.py`

**Interfaces:**

```python
@dataclasses.dataclass(frozen=True)
class StateVar:
    name: str
    dtype: str = "f8"                  # numpy dtype string
    shape: tuple[int, ...] = ()        # per particle: () or (k,)
    fill: float | int = np.nan
    output: bool = True
    units: str = ""
    long_name: str = ""
    kind: str | None = None            # "extensive" | "intensive" | None
    dim: str | None = None             # name of the (k,) dimension
    labels: tuple[str, ...] | None = None
```

- `NetworkSolver.STATE: ClassVar[tuple[StateVar, ...]] = ()`; `__init__` allocates `self._state[name] = np.full((n, *shape), fill, dtype)`; `self.state` is a `Mapping[str, NDArray]` of read-only views; `self.state_specs` returns the tuple. Reserved names (`reach`, `s`, `status`, `mass`, ...) raise `ValueError` at construction.
- `NetworkWriter(..., state_specs: Sequence[StateVar] = ())` creates one variable per spec with `output=True`, dims `("time", "particle")` or `("time", "particle", spec.dim)`, attrs `units`, `long_name`, `kind`, and `fluvial_particle_state = 1`; a `(k,)` spec creates the dimension and a `labels` coordinate variable when given. `write_step(..., state: Mapping[str, NDArray] | None = None)` writes `state[name][lo:hi]`.
- `NetworkResults.state_variables -> tuple[str, ...]` (variables carrying the marker attribute); `positions(time)` adds a column per scalar state variable and one column per label for vector ones; `positions()` includes them in the Dataset.

- [ ] **Step 1: Failing tests**

`test_particles.py`: `StateVar` defaults; a solver subclass with `STATE = (StateVar("zeta"), StateVar("c", shape=(2,), dim="constituent", labels=("a", "b"), kind="extensive"))` allocates `(n,)` and `(n, 2)` arrays filled with NaN; `state["zeta"]` is read-only (`assert not view.flags.writeable`); a reserved name raises.

`test_writer.py`: round trip of both variables through `NetworkWriter` / `xarray.open_dataset`, dims and attrs as specified, `constituent` coordinate labels `["a", "b"]`.

`test_results.py`: `state_variables == ("zeta", "c")`; `positions(0)` has columns `zeta`, `c_a`, `c_b`.

- [ ] **Step 2: Implement**, then run `run.py` through the new signature: `writer = NetworkWriter(..., state_specs=solver.state_specs)` and `writer.write_step(..., state=solver.state)`.

- [ ] **Step 3: Verify, lint, mypy, commit**

```bash
git commit -am "Declared per-particle state: StateVar, solver allocation, writer and results generics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Registry and `[network.particles]`

**Files:**
- Modify: `particles.py` (`PARTICLE_MODELS`, `resolve_model`), `config.py` (`ParticlesConfig`, `NetworkConfig.particles`, template), `solver.py` (`validate_params`, `params` kwarg, `resolves_vertical`), `run.py`, `__init__.py`
- Test: `test_particles.py`, `test_config.py`, `test_run.py`

**Interfaces:**
- `PARTICLE_MODELS: dict[str, type[NetworkSolver]] = {"passive": NetworkSolver}` (drift added in Task 8).
- `resolve_model(name: str) -> type[NetworkSolver]`: registry lookup, or `"pkg.mod:Class"` via `importlib`; the class must subclass `NetworkSolver` or `TypeError`.
- `NetworkSolver.__init__(..., params: Mapping[str, Any] | None = None)`; `NetworkSolver.validate_params(params) -> dict[str, Any]` classmethod: base raises `ValueError` on any key; `self.params = validate_params(params or {})`.
- `NetworkSolver.resolves_vertical: ClassVar[bool] = False`.
- `ParticlesConfig(model: str = "passive", params: Mapping[str, Any] = {})`, frozen; `from_dict` puts every key other than `model` into `params`; `__post_init__` calls `resolve_model(model).validate_params(params)`; `to_dict` flattens back.
- `NetworkConfig.particles: ParticlesConfig` default passive; `from_dict` parses the nested table; unknown keys inside `[network.particles]` are the model's business.
- `run.py`: `cls = resolve_model(cfg.particles.model)`; `cls(..., params=cfg.particles.params)`; attrs `particle_model`, `particles` (JSON).

- [ ] **Step 1: Failing tests**: registry lookups (name, dotted path to a test class in `tests/network/support.py`, unknown name → `KeyError` with the known names in the message, non-subclass → `TypeError`); `ParticlesConfig.from_dict({"model": "passive", "x": 1})` raises via `validate_params`; `NetworkConfig.from_dict` with and without the table; `run_network_simulation` with `particles = {model = "tests.network.support:HalfSpeed"}` writes `particle_model` into the file.

- [ ] **Step 2: Implement.** Template addition (`get_network_config_template`):

```toml
# [network.particles]
# model = "passive"            # or "drift"; or "package.module:Class" subclassing NetworkSolver
# Drift model parameters (model = "drift"):
# settling_velocity = 0.0      # m/s, positive down
# swim_velocity = 0.0          # m/s, positive up
# deposition_velocity = 0.0    # m/s; 0 is a reflecting bed
# critical_ustar = 0.2         # m/s; no deposition where the reach ustar exceeds it
# zeta_min = 0.001             # walk domain [zeta_min, 1 - zeta_min]
# max_substeps = 500
# initial_zeta = "uniform"     # or a number
# [network.particles.diel]    # optional sinusoidal vertical velocity
# amplitude = 0.005            # m/s
# period = 86400.0             # s
# phase = 0.0                  # radians
```

- [ ] **Step 3: Verify, lint, commit**

```bash
git commit -am "Particle model registry and [network.particles]; run wires the configured class

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Dispersion configuration: background, shear correction, vertical sub-table

**Files:**
- Modify: `config.py`, `dispersion.py`, `solver.py` (`_correct_shear`), `__init__.py`
- Test: `test_config.py`, `test_dispersion.py`, `test_solver.py`

**Interfaces:**

```python
VERTICAL_PROFILES = ("parabolic", "constant", "value")
VELOCITY_PROFILES = ("log", "uniform")
SHEAR_CORRECTIONS = ("auto", "on", "off")

@dataclasses.dataclass(frozen=True)
class VerticalDispersionConfig:
    profile: str = "parabolic"
    kappa: float = 0.41
    beta: float = 0.067
    value: float | None = None
    background: float = 0.0
    scale: float = 1.0
    velocity_profile: str = "log"
```

- `DispersionConfig` gains `background: float = 0.0`, `shear_correction: str = "auto"`, `vertical: VerticalDispersionConfig = default`; `from_dict` parses the nested `vertical` mapping; `to_dict` nests it (JSON-safe).
- `dispersion_coefficient(..., background=0.0)`: adds `background` where `flow_out > 0` for every model, including `"none"` (a user asking for a background K on top of no model gets it).
- `NetworkSolver._correct_shear(k, h) -> k`: identity when `self._shear_table is None`. The table is built in Task 7; here only the resolution rule is implemented:

```python
    def _shear_correction_active(self) -> bool:
        mode = self.dispersion.shear_correction
        if mode == "off":
            return False
        if mode == "on":
            if not self.resolves_vertical:
                raise ValueError("dispersion.shear_correction = 'on' requires a particle model that resolves the vertical")
            return True
        return self.resolves_vertical and self.dispersion.vertical.velocity_profile != "uniform"
```

- [ ] **Step 1: Failing tests**: validation of each field (profile names, `value` required for `"value"`, positive `kappa`/`beta`/`scale`, non-negative backgrounds); nested `from_dict` / `to_dict` round trip; `background` added in `dispersion_coefficient` for all three models and zero on dry reaches; `_shear_correction_active()` truth table over (`mode`, `resolves_vertical`, `velocity_profile`) using two tiny subclasses; `"on"` on the passive model raises at construction.

- [ ] **Step 2: Implement; verify; lint; commit**

```bash
git commit -am "Dispersion config: background term, shear_correction switch, vertical sub-table

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `vertical.py`: profiles, walk, sub-steps, Taylor integral, deposition probability

**Files:**
- Create: `src/fluvial_particle/network/vertical.py`
- Test: `tests/network/test_vertical.py`

**Interfaces:**

```python
class VerticalProfiles:
    """Velocity factor and vertical mixing profiles for one VerticalDispersionConfig and zeta_min."""

    def __init__(self, cfg: VerticalDispersionConfig, zeta_min: float, *, n_ratio: int = 65, n_quad: int = 20001) -> None: ...
    # tabulated on ratio = ustar / v in [0, 2] at construction:
    #   self._norm[r]  : mean over the domain of max(1 + r (1 + ln zeta) / kappa, 0)
    #   self._shear[r] : Taylor coefficient c(r) on the domain with the floored, renormalized factor
    def velocity_factor(self, zeta: FloatArray, ustar: FloatArray, v: FloatArray) -> FloatArray: ...
    def kz(self, zeta: FloatArray, ustar: FloatArray, h: FloatArray) -> FloatArray: ...
    def dkz_dz(self, zeta: FloatArray, ustar: FloatArray, h: FloatArray) -> FloatArray: ...
    def kz_max(self, ustar: FloatArray, h: FloatArray) -> FloatArray: ...
    def shear_coefficient(self, ustar: FloatArray, v: FloatArray) -> FloatArray: ...   # c per reach, 0 where v == 0

def substep_count(dt: float, kz_max: FloatArray, h: FloatArray, *, c: float = 0.1, max_substeps: int = 500) -> int: ...
def deposition_probability(k_d: FloatArray, dt_sub: float, kz_bed: FloatArray) -> FloatArray: ...   # clip(k_d sqrt(pi dt_sub / kz_bed), 0, 1)
def shear_dispersion_coefficient(cfg: VerticalDispersionConfig, zeta_min: float, ratio: float | None = None, *, n: int = 200001) -> float: ...
```

Formulas (spec Decision 7 and 7a; `g(zeta) = (1 + ln zeta) / kappa` is the log-law deviation in units of `ustar`):

- log law: raw factor `1 + (ustar / v) g(zeta)`, floored at 0, divided by `norm(ustar / v)` so its mean over the domain is exactly 1; `"uniform"`: 1.
- `"parabolic"`: `Kz = scale * kappa * ustar * h * zeta (1 - zeta) + background`, `dKz/dz = scale * kappa * ustar * (1 - 2 zeta)`, `Kz_max = scale * kappa * ustar * h / 4 + background`.
- `"constant"`: `Kz = scale * beta * ustar * h + background`, `dKz/dz = 0`.
- `"value"`: `Kz = scale * value + background`, `dKz/dz = 0`.
- Taylor coefficient on the domain `[lo, hi]` with `q = Kz / (ustar h)` dimensionless: `g` the (floored, renormalized when `ratio` is given) factor deviation with its domain mean removed, `G1 = cumint(g)`, `G2 = cumint(G1 / q)`, `c = -int(g * G2)`. For `"value"` there is no `ustar h` scaling; return `c` such that `K_shear = c * ustar * h` still, using `q = value / (ustar h)` per reach: tabulate on both `ratio` and `ustar h / value`? Simpler and sufficient: for `"value"` compute per reach at step time on a coarse grid (`n_quad = 2001`), vectorized over reaches; it is the rare profile. State this in the docstring.
- Deposition probability: `p = k_d sqrt(pi dt_sub / Kz(zeta_min))` clipped to `[0, 1]` (Erban and Chapman 2007); `Kz(zeta_min) > 0` is guaranteed when `background > 0` or `zeta_min > 0` for the parabolic profile.

- [ ] **Step 1: Failing tests**

```python
def test_log_law_factor_has_unit_mean_after_floor_and_normalization():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    z = np.linspace(0.001, 0.999, 200001)
    f = vp.velocity_factor(z, np.full_like(z, 0.1), np.full_like(z, 0.7))
    assert f.min() >= 0.0
    assert np.trapezoid(f, z) / (z[-1] - z[0]) == pytest.approx(1.0, abs=1e-4)

def test_parabolic_kz_and_gradient():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    z = np.array([0.0, 0.25, 0.5, 1.0]); u = np.full(4, 0.1); h = np.full(4, 2.0)
    np.testing.assert_allclose(vp.kz(z, u, h), 0.41 * 0.1 * 2.0 * z * (1 - z))
    np.testing.assert_allclose(vp.dkz_dz(z, u, h), 0.41 * 0.1 * (1 - 2 * z))
    assert vp.kz_max(u, h)[0] == pytest.approx(0.41 * 0.1 * 2.0 / 4)

def test_constant_profile_matches_parabolic_depth_mean_at_defaults():
    # kappa / 6 = 0.0683 vs beta = 0.067
    ...

def test_taylor_coefficient_reproduces_elder_and_truncation_values():
    cfg = VerticalDispersionConfig()
    assert shear_dispersion_coefficient(cfg, zeta_min=0.0) == pytest.approx(0.404 / 0.41**3, rel=5e-3)   # 5.86
    assert shear_dispersion_coefficient(cfg, zeta_min=0.01) == pytest.approx(4.53, rel=1e-2)
    assert shear_dispersion_coefficient(cfg, zeta_min=0.001) == pytest.approx(5.65, rel=1e-2)
    assert shear_dispersion_coefficient(cfg, zeta_min=0.0, ratio=0.143) == pytest.approx(5.23, rel=1e-2)
    const = VerticalDispersionConfig(profile="constant")
    assert shear_dispersion_coefficient(const, zeta_min=0.0) == pytest.approx(6.58, rel=1e-2)
    assert shear_dispersion_coefficient(VerticalDispersionConfig(velocity_profile="uniform"), zeta_min=0.001) == 0.0

def test_shear_table_interpolates_the_quadrature():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    c = vp.shear_coefficient(np.array([0.1, 0.0]), np.array([0.7, 0.5]))
    assert c[0] == pytest.approx(shear_dispersion_coefficient(VerticalDispersionConfig(), 0.001, ratio=0.1 / 0.7), rel=1e-2)
    assert c[1] == 0.0

def test_substep_count_at_drb_medians_and_cap():
    n = substep_count(900.0, kz_max=np.array([0.41 * 0.1 * 0.59 / 4]), h=np.array([0.59]))
    assert 140 <= n <= 170
    assert substep_count(900.0, kz_max=np.array([10.0]), h=np.array([0.1]), max_substeps=500) == 500

def test_deposition_probability_formula_and_clip():
    p = deposition_probability(np.array([1e-4, 1.0]), 5.0, np.array([1e-4, 1e-4]))
    assert p[0] == pytest.approx(1e-4 * np.sqrt(np.pi * 5.0 / 1e-4))
    assert p[1] == 1.0
```

The reference numbers were produced by the quadrature in the spec work (2026-09-16) with a 200,001-point trapezoid on `[max(zeta_min, 1e-9), 1 - max(zeta_min, 1e-9)]`; if the implementation's quadrature differs (Simpson, more points), the values move in the third digit and the tolerances above still hold.

- [ ] **Step 2: Implement.** Keep every function pure (no solver state). Use `scipy.integrate.cumulative_trapezoid` for the cumulative integrals. Guard `q` at the domain ends by using `[max(zeta_min, 1e-9), 1 - max(zeta_min, 1e-9)]`.

- [ ] **Step 3: Verify, lint, mypy, commit**

```bash
git commit -am "vertical.py: velocity and mixing profiles, Taylor shear coefficient, sub-steps, deposition probability

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `DriftParticles`

**Files:**
- Modify: `particles.py` (add the class and register `"drift"`), `solver.py` (`_correct_shear` uses `VerticalProfiles.shear_coefficient` when the model provides one), `run.py` (diagnostics)
- Test: `tests/network/test_particles.py`

**Interfaces:**

```python
class DriftParticles(NetworkSolver):
    resolves_vertical = True
    STATE = (
        StateVar("zeta", units="1", long_name="relative elevation in the column, 0 bed, 1 surface"),
        StateVar("velocity_factor", units="1", long_name="mean of u(zeta)/v over the step's substeps", output=False),
    )
    PARAM_DEFAULTS = {
        "settling_velocity": 0.0, "swim_velocity": 0.0, "diel": None,
        "deposition_velocity": 0.0, "critical_ustar": None,
        "zeta_min": 0.001, "max_substeps": 500, "initial_zeta": "uniform",
    }

    @classmethod
    def validate_params(cls, params): ...   # unknown key raises; ranges; diel table keys amplitude/period/phase

    def __init__(self, *args, params=None, **kwargs):
        super().__init__(*args, params=params, **kwargs)
        self.profiles = VerticalProfiles(self.dispersion.vertical, self.params["zeta_min"])
        self._shear_table = self.profiles if self._shear_correction_active() else None

    def on_release(self, idx, h):
        lo, hi = self.params["zeta_min"], 1 - self.params["zeta_min"]
        init = self.params["initial_zeta"]
        self._state["zeta"][idx] = self.rng.uniform(lo, hi, idx.size) if init == "uniform" else float(init)

    def behave(self, h, tau, t, dt):
        idx = np.nonzero(self._status == ACTIVE)[0]
        if idx.size == 0:
            return None
        r = self._reach[idx].astype(np.int64)
        ustar, depth, v = (np.asarray(h[k], dtype=np.float64)[r] for k in ("ustar", "depth", "velocity"))
        zmin = self.params["zeta_min"]; lo, hi = zmin, 1.0 - zmin
        kzmax = self.profiles.kz_max(ustar, depth)
        n_sub = substep_count(dt, kzmax, depth, max_substeps=self.params["max_substeps"])
        self.last_substeps = n_sub
        w = self._vertical_velocity(t + 0.5 * dt)                   # positive down, scalar or per particle
        k_d = self._deposition_velocity(ustar)                        # Krone factor applied
        zeta = self._state["zeta"][idx].copy()
        fsum = np.zeros(idx.size)
        alive = np.ones(idx.size, dtype=bool)
        dt_sub = dt / n_sub
        # particles released mid-step walk only for tau; use per-particle sub-step lengths tau / n_sub
        dts = tau[idx] / n_sub
        for _ in range(n_sub):
            a = alive
            kz = self.profiles.kz(zeta[a], ustar[a], depth[a])
            drift = (self.profiles.dkz_dz(zeta[a], ustar[a], depth[a]) - w_a) / depth[a]
            zeta_new = zeta[a] + drift * dts[a] + np.sqrt(2.0 * kz * dts[a] / depth[a] ** 2) * self.rng.standard_normal(a.sum())
            # deposition check before reflection at the bed
            below = zeta_new < lo
            if below.any() and np.any(k_d[a] > 0):
                p = deposition_probability(k_d[a][below], dts[a][below], self.profiles.kz(np.full(below.sum(), lo), ustar[a][below], depth[a][below]))
                deposit = self.rng.uniform(size=below.sum()) < p
                ...  # mark those in `alive` False, zeta = lo, and collect for terminate()
            zeta[a] = reflect_interval(zeta_new, lo, hi)
            fsum[a] += self.profiles.velocity_factor(zeta[a], ustar[a], v[a])
        self._state["zeta"][idx] = zeta
        factor = np.ones(self.n)
        factor[idx[alive]] = fsum[alive] / n_sub
        if (~alive).any():
            self.terminate(idx[~alive], SETTLED, t + dt)
        self._state["velocity_factor"][idx] = factor[idx]
        return factor
```

Details the implementation must keep:
- The mean factor is over the substeps the particle was alive; deposited particles are terminated and excluded from advection (their `s` stays where the step began, a first-order timing approximation noted in the docs).
- `_vertical_velocity(t_mid) = settling_velocity - swim_velocity + amplitude * sin(2 pi t_mid / period + phase)` (positive down).
- `_deposition_velocity(ustar) = deposition_velocity * max(0, 1 - (ustar / critical_ustar)^2)` when `critical_ustar` is set, else the constant.
- `zeta` of a deposited particle is set to `zeta_min`, of an exited particle left as is (it is finite and harmless; `positions` shows it for the last active time).
- A zero-velocity reach (`v == 0`) has `ustar == 0`; `kz` is then `background` only, `n_sub` from `background`, the factor 1 (uniform), and `kz(zeta_min) == 0` when `background == 0` makes `deposition_probability` return 0 (guard the division: probability 0 where `kz_bed == 0`).

`solver.py`:

```python
    def _correct_shear(self, k, h):
        if self._shear_table is None:
            return k
        c = self._shear_table.shear_coefficient(np.asarray(h["ustar"]), np.asarray(h["velocity"]))
        return np.maximum(k - c * np.asarray(h["ustar"]) * np.asarray(h["depth"]), 0.0)
```

with `self._shear_table = None` set in the base `__init__` before subclasses run.

`run.py` diagnostics: when `solver.resolves_vertical`, append to the startup report the sub-step count at the median reach for the first step (`substep_count` on the first hydraulics slice), the number of reaches that would hit `max_substeps`, and whether the shear correction is active with its coefficient range.

- [ ] **Step 1: Failing tests** (`test_particles.py`)

- `validate_params`: defaults filled, unknown key raises, negative `deposition_velocity` raises, `initial_zeta` outside the domain raises, `diel` missing a key raises.
- `on_release` puts `zeta` inside `[zeta_min, 1 - zeta_min]`, fixed value when a number.
- `zeta` is preserved across a reach hop: two reaches, `dispersion.vertical.profile = "value"`, `value = 0`, `background = 0` (no walk), `velocity_profile = "uniform"`, fixed `initial_zeta = 0.3`; after a hop `zeta[0] == 0.3`.
- Velocity factor: `initial_zeta = 0.5`, no mixing (as above but `velocity_profile = "log"`): the landing `s` equals `f(0.5) * v * dt` with `f` from `VerticalProfiles`.
- Deposition: `deposition_velocity = 1e9`, `settling_velocity = 0.05`, shallow reach: after one step every particle is `SETTLED`, keeps a finite `s`, `exit_reach == reach`, `zeta == zeta_min`; with `critical_ustar` below the reach `ustar` none settle.
- Terminal exclusion: settled particles do not advance on the next step.
- Shear correction active: `DriftParticles` with default dispersion has `_shear_table is not None`; with `velocity_profile = "uniform"` it is `None`; passive with `"on"` raises (Task 6 test moves here if not already present).
- RNG: a passive run's arrays are unchanged by the existence of the drift class (import order and registry do not touch the base).

- [ ] **Step 2: Implement; verify; lint; mypy; commit**

```bash
git commit -am "DriftParticles: sub-stepped vertical walk, log-law velocity factor, Robin deposition; shear correction of K

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Analytical acceptance tests

**Files:**
- Modify: `tests/network/test_analytical.py` (all `@pytest.mark.slow`), `tests/network/support.py` (a `uniform_reach_dataset(length, velocity, depth, ustar, width, n_reach)` helper if `chain_dataset` cannot set `ustar` and `depth` independently of the Fischer target)

Runs use `ArrayHydraulicsProvider`, one long reach or a 10-reach chain, `dispersion.model = "none"` unless stated, `dt = 60 s`, 20,000 particles released as a slug at `s = 0`, seeded.

- [ ] **Test 1: Uniform stays uniform.** Drift model, `w = 0`, `deposition_velocity = 0`, `initial_zeta = "uniform"`, one reach, 1 h: chi-square of the `zeta` histogram (20 equal bins over the domain) against uniform, `p > 0.01`. Then the negative control: monkeypatch `VerticalProfiles.dkz_dz` to return zeros and assert `p < 1e-6` (the drift term is load-bearing).

- [ ] **Test 2: Rouse profile.** `settling_velocity = P * kappa * ustar` for `P = 0.5` and `P = 2`, reflecting bed, 3 h on a long reach, then compare `zeta` to the Rouse distribution truncated to the domain by Kolmogorov-Smirnov with the CDF obtained by numerical integration of `((1 - z) / z * a / (1 - a))**P` with `a = zeta_min`; `p > 0.01`.

- [ ] **Test 3: Mean advection preserved.** Passive drift (`w = 0`), `dispersion.model = "none"`, uniform initial `zeta`, 6 h on a 50 km reach: mean `s` within 3 standard errors of `v t`.

- [ ] **Test 4: Shear dispersion emerges.** Same run: `var(s)` at `t = 6 h` against `2 c ustar h t` with `c = VerticalProfiles(...).shear_coefficient(ustar, v)` for the run's `zeta_min` and `ustar / v`, within 10 percent (the Taylor regime needs `t >> h^2 / Kz`, satisfied by orders of magnitude). Repeat with `profile = "constant"`. Then with `dispersion.model = "fischer"` and `shear_correction = "auto"`: total variance against `2 K_fischer t` within 10 percent, which shows the correction removes exactly the emergent part.

- [ ] **Test 7: Deposition against the Robin condition.** `settling_velocity = 0.01`, `deposition_velocity = 1e-3`, uniform release, reflecting surface, on a reach with `ustar = 0.1`, `depth = 1`: the deposited fraction at `t = 10, 20, 30 min` against a Crank-Nicolson finite-difference solution of `dC/dt = d/dz (Kz dC/dz) + w dC/dz` with `-Kz dC/dz - w C = k_d C` at `z = zeta_min h` and zero flux at the surface (200 cells, dt 1 s), within 0.02 absolute. Repeat with `max_substeps = 40` (halving the count at these hydraulics): the deposited fraction moves by less than 0.01. Then `deposition_velocity = 1e6`: against the absorbing-bed solution.

- [ ] **Verify** `conda run -n fluvial-particle pytest tests/network/test_analytical.py -m slow -q` and **commit**

```bash
git commit -am "Analytical acceptance tests for the drift model: uniformity, Rouse, mean advection, shear dispersion, Robin deposition

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Documentation, changelog, rules

**Files:**
- Modify: `docs/network.rst` (new sections), `HISTORY.md`, `.claude/rules/network.md`, `src/fluvial_particle/network/config.py` (template already in Task 5)

- [ ] `docs/network.rst`: a "Behavioral particles" section after "Sources": the model table and how to name a class; the drift model with the vertical convention, the walk, the sub-step rule and its cost, the deposition parameters with their physical reading (deposition velocity, Krone critical shear), the diel option; a "Dispersion configuration" section listing the `[network.dispersion]` keys with the `vertical` sub-table and the parallel to the 2D/3D `lev` and `beta`; a "Shear dispersion and the Fischer coefficient" paragraph stating the double count, the DRB and narrow-stream numbers, the truncation effect (5.86 full column, 5.65 at 0.001, 4.53 at 0.01), and that the correction is computed on the walk's domain; an "Approximations" addition for the deposited particle's `s` being the step's start and for the vertical walk resolving nothing when `n_sub` hits the cap.
- [ ] `HISTORY.md` under Unreleased, "New features": one entry summarizing the registry, hooks, declared state, terminal statuses, drift model, dispersion configuration, and that the passive model is unchanged.
- [ ] `.claude/rules/network.md`: add rows for `particles.py`, `vertical.py`, and `random_walk.py`; add the rules "models declare state and never touch the writer", "the same floored, renormalized velocity factor and the same truncated domain are used by the walk and by the shear quadrature", "vertical velocity is positive down".
- [ ] Build the docs (`conda run -n fluvial-particle sphinx-build -W docs docs/_build/html`) and commit.

```bash
git commit -am "Document behavioral particles, the vertical model, and dispersion configuration

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Final review and PR

- [ ] `conda run -n fluvial-particle ruff check . && conda run -n fluvial-particle ruff format --check .`
- [ ] `conda run -n fluvial-particle mypy src/`
- [ ] `conda run -n fluvial-particle pytest -q` (all, including slow) and `conda run -n fluvial-particle nox -s "tests-3.13(vtk97)"` so the 2D/3D reflection extraction is exercised on both VTK versions.
- [ ] If `FLUVIAL_PARTICLE_DRB_FILE` is set: run the DRB smoke test, then a one-day drift-model run from the headwaters (`settling_velocity = 0.005`, `deposition_velocity = 1e-4`, `critical_ustar = 0.15`) and check the startup diagnostics report a sensible sub-step count and that some particles settle.
- [ ] Rebase onto `main` if #53 has merged; otherwise open the PR against `spec/network-behavioral-particles` and retarget on merge.
- [ ] `/pr-review-toolkit:review-pr` before requesting merge.

---

## Not in this plan

- Decisions 9 to 11 (decay, temperature, water seeding with age and source fraction, `concentration(variable=)`): PR 2.
- Decision 12 (reactor operator): extension point.
- Brownian-bridge exits, LaBolle correction across longitudinal `K` jumps, distributaries: base spec extension points.

## Self-review notes

- **Spec coverage:** Decision 1 (Task 5), 2 (Task 3), 3 (Task 4), 4 partially (mass changes are allowed by `behave`; `kind` is written; `concentration(variable=)` is PR 2), 5 (Task 2), 6 (nothing imports bins; Task 4 `state` is the write surface a reactor will use), 7 and 7a (Tasks 6 to 8), 8 (Task 8); analytical tests 1 to 4 and 7 (Task 9); docs (Task 10).
- **Bit-identity of the passive model:** Task 3 keeps `1.0 * v` arithmetic and the RNG sequence; Task 4 allocates nothing for an empty `STATE`; Task 6's `background = 0.0` adds zero; `_correct_shear` is identity when `_shear_table is None`. The whole existing `tests/network` suite is the check, run at every task.
- **Consistency of the shear correction:** `VerticalProfiles` owns both the walk's velocity factor and the quadrature; `shear_dispersion_coefficient` is a thin wrapper that builds the same tables, so the two cannot drift apart.
- **Quadrature reference numbers** (5.86, 5.65, 4.53, 5.23, 6.58) come from a 200,001-point trapezoid computed during spec review on 2026-09-16; Task 7 tests use 0.5 to 1 percent tolerances.
- **Type consistency:** `behave` returns `FloatArray | None`; `_advect(v, tau, t, dt, factor)`; `write_step(itime, time_seconds, reach, s, status, lo, hi, state=None)`; `NetworkWriter(..., state_specs=())`; `resolve_model(name) -> type[NetworkSolver]`; `cls(..., params=...)` accepted by every model.
