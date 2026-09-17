"""Vectorized 1D network particle solver: exact advection with time carry, then one dispersive kick."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

import numpy as np
import numpy.typing as npt

from .dispersion import dispersion_coefficient
from .network import Network
from .provider import HydraulicsProvider
from .sources import ParticleSchedule


if TYPE_CHECKING:
    # Type-only: config.py imports particles.py (for the registry), which imports this module.
    from .config import DispersionConfig
    from .particles import StateVar


class ShearTable(Protocol):
    """What the shear correction needs from a model: Taylor's coefficient per reach."""

    def shear_coefficient(self, ustar: FloatArray, v: FloatArray, h: FloatArray | None = None) -> FloatArray:
        """Dimensionless shear-dispersion coefficient ``c`` per reach (``K_shear = c ustar h``)."""
        ...


class Status(IntEnum):
    """Particle status codes, stored as int8 in the state array and in the output file."""

    UNRELEASED = 0
    ACTIVE = 1
    EXITED = 2  # left the network at an outlet
    SETTLED = 3  # deposited on the bed (behavioral models)
    REMOVED = 4  # taken out by a model (mass floor, mortality)


# Module-level aliases: the codes read as bare names in the solver and in user code.
UNRELEASED = Status.UNRELEASED
ACTIVE = Status.ACTIVE
EXITED = Status.EXITED
SETTLED = Status.SETTLED
REMOVED = Status.REMOVED
# Every code above ACTIVE is terminal: the particle no longer moves and exit_time/exit_reach are set.
TERMINAL_STATUSES = tuple(code for code in Status if code > Status.ACTIVE)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
# A step's hydraulics dict as the provider returns it (export names; float64 or float32 arrays).
Hydraulics = Mapping[str, npt.NDArray[np.floating[Any]]]
# Names the base solver and the writer own; a model's declared state may not reuse them.
RESERVED_NAMES = frozenset({
    "reach",
    "s",
    "prev_reach",
    "status",
    "mass",
    "exit_time",
    "exit_reach",
    "reach_index",
    "reach_id",
    "time",
    "time_seconds",
    "source_index",
    "release_reach",
    "release_s",
    "release_time",
    "particle",
})


def _readonly(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """A non-writable view of ``arr`` that tracks the backing array's contents (guards accidental writes)."""
    view = arr.view()
    view.setflags(write=False)
    return view


class NetworkSolver:
    """Particle state arrays and the per-step update; the base class and the passive model.

    Subclasses declare per-particle state in ``STATE`` and override the ``on_release`` and ``behave``
    hooks; the transport kernel (release, advection with time carry, one dispersive kick) is shared.

    Args:
        network: static topology.
        provider: per-step hydraulics.
        schedule: this solver's particles (already sliced for the MPI rank).
        start_time: run start; the solver clock is seconds from it.
        dt: step (s).
        dispersion: dispersion settings.
        rng: random state supplying standard normals.
        max_hops: cap on reach hops per particle per step; exceeding it raises RuntimeError.
        params: the model's parameters from ``[network.particles]`` (everything but ``model``);
            validated and completed with defaults by ``validate_params``.
    """

    STATE: ClassVar[tuple[StateVar, ...]] = ()
    # True for models that track a vertical position; drives the shear correction of the
    # longitudinal dispersion coefficient (see DispersionConfig.shear_correction).
    resolves_vertical: ClassVar[bool] = False

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
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Allocate particle state arrays from the schedule and prepare the solver clock.

        Args:
            network: static topology.
            provider: per-step hydraulics.
            schedule: this solver's particles (already sliced for the MPI rank).
            start_time: run start; the solver clock is seconds from it.
            dt: step (s).
            dispersion: dispersion settings.
            rng: random state supplying standard normals.
            max_hops: cap on reach hops per particle per step; exceeding it raises RuntimeError.
            params: model parameters (see ``validate_params``).
        """
        self.params: dict[str, Any] = self.validate_params(params or {})
        self.network = network
        self.provider = provider
        self.start_time: np.datetime64 = cast("np.datetime64", start_time.astype("datetime64[ns]"))
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
        self._reach: npt.NDArray[np.int32] = np.full(n, -1, dtype=np.int32)
        self._s: npt.NDArray[Any] = np.full(n, np.nan, dtype=fdtype)
        self._prev_reach: npt.NDArray[np.int32] = np.full(n, -1, dtype=np.int32)
        self._status: npt.NDArray[np.int8] = np.full(n, UNRELEASED, dtype=np.int8)
        self._exit_time: FloatArray = np.full(n, np.nan)
        self._exit_reach: npt.NDArray[np.int32] = np.full(n, -1, dtype=np.int32)
        self._views: dict[str, npt.NDArray[Any]] = {
            name: _readonly(getattr(self, f"_{name}"))
            for name in ("reach", "s", "prev_reach", "status", "exit_time", "exit_reach")
        }
        self._length = network.length
        self._to_index = network.to_index.astype(np.int64)
        self._state: dict[str, npt.NDArray[Any]] = self._allocate_state(n)
        self._state_views: Mapping[str, npt.NDArray[Any]] = MappingProxyType({
            name: _readonly(arr) for name, arr in self._state.items()
        })
        # Set by models that resolve the vertical to the object whose shear_coefficient(ustar, v, h)
        # gives Taylor's coefficient per reach; None leaves the longitudinal K uncorrected.
        self._shear_table: ShearTable | None = None
        self.last_shear_zeroed = 0  # reaches whose longitudinal K the correction took to 0 in the last step
        self._warned: set[str] = set()  # conditions already reported by _warn_once
        self._shear_correction_active()  # "on" with a model that cannot honour it is a config error

    def _allocate_state(self, n: int) -> dict[str, npt.NDArray[Any]]:
        """Allocate one array per declared ``STATE`` entry, validating the declarations.

        Args:
            n: particle count.

        Returns:
            The state arrays keyed by name, in declaration order.

        Raises:
            ValueError: a name (or vector dim) the solver and the output file reserve, a duplicate
                name, or two states sharing a dim with a different shape or labels.
        """
        state: dict[str, npt.NDArray[Any]] = {}
        dims: dict[str, StateVar] = {}
        for spec in self.STATE:
            if spec.name in RESERVED_NAMES:
                raise ValueError(f"state name {spec.name!r} is reserved by the solver or the output file")
            if spec.name in state:
                raise ValueError(f"duplicate state name {spec.name!r} in {type(self).__name__}.STATE")
            if spec.dim is not None:
                if spec.dim in RESERVED_NAMES:
                    raise ValueError(f"state {spec.name!r}: dim {spec.dim!r} is a name the output file already uses")
                first = dims.setdefault(spec.dim, spec)
                if (first.shape, first.labels) != (spec.shape, spec.labels):
                    raise ValueError(
                        f"states {first.name!r} and {spec.name!r} share dim {spec.dim!r} with different shape or labels"
                    )
            state[spec.name] = np.full((n, *spec.shape), spec.fill, dtype=np.dtype(spec.dtype))
        return state

    def _warn_once(self, key: str, message: str) -> None:
        """Report ``message`` as a UserWarning the first time ``key`` comes up in this run.

        Args:
            key: names the condition; later reports of the same key are silent.
            message: the warning text. ``stacklevel`` points at the caller of the method reporting it
                (``step`` for a condition a model or the kernel meets during a step).
        """
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _shear_correction_active(self) -> bool:
        """Resolve ``dispersion.shear_correction`` for this model (see DispersionConfig).

        With ``dispersion.model = "none"`` there is no longitudinal K to correct, so the correction is
        never active (the resolved vertical shear is then the run's only longitudinal dispersion).

        Raises:
            ValueError: ``"on"`` with a model that does not resolve the vertical.
        """
        mode = self.dispersion.shear_correction
        if mode == "off":
            return False
        if mode == "on":
            if not self.resolves_vertical:
                raise ValueError(
                    "dispersion.shear_correction = 'on' requires a particle model that resolves the vertical"
                )
            return self.dispersion.model != "none"
        return (
            self.resolves_vertical
            and self.dispersion.vertical.velocity_profile != "uniform"
            and self.dispersion.model != "none"
        )

    def _correct_shear(self, k: FloatArray, h: Hydraulics) -> FloatArray:
        """Remove the shear-dispersion part ``c ustar h`` from ``k`` when a model resolves the vertical."""
        if self._shear_table is None:
            return k
        ustar = np.asarray(h["ustar"], dtype=np.float64)
        depth = np.asarray(h["depth"], dtype=np.float64)
        c = self._shear_table.shear_coefficient(ustar, np.asarray(h["velocity"], dtype=np.float64), depth)
        shear = c * ustar * depth
        zeroed = (shear > 0.0) & (k < shear)
        self.last_shear_zeroed = int(zeroed.sum())
        if self.last_shear_zeroed:
            self._warn_once(
                "shear_zeroed",
                f"the resolved vertical shear exceeds the longitudinal K on {self.last_shear_zeroed} reaches "
                f"(first: index {int(np.nonzero(zeroed)[0][0])}); their K is set to 0 there. The vertical walk "
                "generates more shear dispersion than the longitudinal model claims exists.",
            )
        return np.maximum(k - shear, 0.0)

    @classmethod
    def validate_params(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the model's parameters and fill in defaults.

        The passive model takes none. Subclasses override this to declare theirs; an unknown key
        always raises, as everywhere in the config.

        Args:
            params: the ``[network.particles]`` table without ``model``.

        Returns:
            The validated parameters with defaults filled in.

        Raises:
            ValueError: a key the model does not know.
        """
        if params:
            raise ValueError(f"{cls.__name__} takes no parameters, got {sorted(params)}")
        return {}

    # ---- declared model state ---------------------------------------------------
    @property
    def state_specs(self) -> tuple[StateVar, ...]:
        """The model's declared per-particle state, in declaration order."""
        return self.STATE

    @property
    def state(self) -> Mapping[str, npt.NDArray[Any]]:
        """Read-only views of the declared state arrays, keyed by name (shape ``(n, *spec.shape)``).

        Models write through ``self._state[name][...]`` in place; rebinding an entry would detach the view.
        """
        return self._state_views

    # ---- particle state (read-only views of the solver's own arrays) --------
    @property
    def reach(self) -> npt.NDArray[np.int32]:
        """Reach index of each particle, -1 when unreleased or exited (particles terminated by a model keep theirs)."""
        return self._views["reach"]

    @property
    def s(self) -> npt.NDArray[Any]:
        """Distance from the upstream end of ``reach`` (m), NaN when unreleased or exited (terminated by a model: kept)."""
        return self._views["s"]

    @property
    def prev_reach(self) -> npt.NDArray[np.int32]:
        """Reach each particle most recently arrived from, -1 when there is no history."""
        return self._views["prev_reach"]

    @property
    def status(self) -> npt.NDArray[np.int8]:
        """`Status` code of each particle."""
        return self._views["status"]

    @property
    def exit_time(self) -> FloatArray:
        """Seconds from the run start at which each particle reached a terminal status, NaN if it has not."""
        return self._views["exit_time"]

    @property
    def exit_reach(self) -> npt.NDArray[np.int32]:
        """Reach at the terminal status (the outlet for exited, the bed reach for settled), -1 otherwise."""
        return self._views["exit_reach"]

    def terminate(self, idx: IntArray, status: int, t_end: float | FloatArray) -> None:
        """Give active particles ``idx`` a terminal status at time ``t_end``; they keep their reach and ``s``.

        Used by behavioral models (a settled particle has a position on the bed). The base solver's
        exits keep their own inline bookkeeping in ``_advect`` and ``_disperse``.

        Args:
            idx: particle indices; every one must be active.
            status: one of ``TERMINAL_STATUSES``.
            t_end: solver time (s) at which the particles reached the status, a scalar or one per particle.

        Raises:
            ValueError: ``status`` is not terminal, or a particle in ``idx`` is not active.
        """
        if status not in TERMINAL_STATUSES:
            raise ValueError(
                f"terminate needs a terminal status {tuple(int(s) for s in TERMINAL_STATUSES)}, got {status}"
            )
        if idx.size == 0:
            return
        if not np.all(self._status[idx] == ACTIVE):
            raise ValueError("terminate: every particle must be active (a terminal or unreleased one was given)")
        self._status[idx] = status
        self._exit_time[idx] = t_end
        self._exit_reach[idx] = self._reach[idx]

    def _exit_at_outlet(self, e: IntArray, reach: IntArray, t_end: float | FloatArray) -> None:
        """Mark particles ``e`` EXITED from ``reach`` at ``t_end`` and clear their position.

        The kernel's own exits, as opposed to a model's ``terminate``: an exited particle keeps no
        reach or ``s``, having left the network.

        Args:
            e: indices of the particles leaving the network.
            reach: the outlet reach each of them left from.
            t_end: solver time (s) of the exit, a scalar or one per particle.
        """
        self._status[e] = EXITED
        self._exit_time[e] = t_end
        self._exit_reach[e] = reach
        self._reach[e] = -1
        self._s[e] = np.nan

    def midpoint_time(self) -> np.datetime64:
        """Datetime at the middle of the step about to be taken."""
        return self.start_time + np.timedelta64(round((self.time + 0.5 * self.dt) * 1e9), "ns")

    def distance_along(self, cum_before: FloatArray) -> FloatArray:
        """Distance from the network origin for active particles: cum_before[reach] + s (NaN otherwise)."""
        out = np.full(self.n, np.nan)
        act = self._status == ACTIVE
        out[act] = cum_before[self._reach[act]] + self._s[act]
        return out

    def step(self) -> None:
        """Advance the clock by dt: release, behave, advect with time carry, disperse with displacement carry.

        The transport kernel is a template method. Models override ``on_release`` and ``behave``; the
        base versions are no-ops, so the passive model is exactly the kernel.
        """
        t = self.time
        dt = self.dt
        h = self.provider.hydraulics(self.midpoint_time())
        tau = self._release(t, dt, h)
        factor = self.behave(h, tau, t, dt)
        if factor is not None:
            self._check_factor(factor)
        v = np.asarray(h["velocity"], dtype=np.float64)
        k = self._dispersion_k(h)
        self._advect(v, tau, t, dt, factor)
        self._disperse(k, tau, t, dt, np.asarray(h["flow_out"], dtype=np.float64))
        self.time = t + dt

    def _dispersion_k(self, h: Hydraulics) -> FloatArray:
        """This step's longitudinal K per reach, shear-corrected when the model resolves the vertical.

        Args:
            h: the step's hydraulics dict (export names).

        Returns:
            K per reach (m^2/s).
        """
        d = self.dispersion
        if self._shear_table is None:
            return dispersion_coefficient(h, d.model, scale=d.scale, cap=d.cap, value=d.value, background=d.background)
        # Correct the model's K, then add the background so it stays the floor it is documented as.
        k = dispersion_coefficient(h, d.model, scale=d.scale, cap=d.cap, value=d.value)
        k = self._correct_shear(k, h)
        if d.background > 0.0:
            k += np.where(np.asarray(h["flow_out"], dtype=np.float64) > 0.0, d.background, 0.0)
        return k

    def _check_factor(self, factor: FloatArray) -> None:
        """Reject a velocity factor of the wrong length or with a non-finite entry on an active particle.

        Raises:
            ValueError: the factor is not a length-``n`` array.
            RuntimeError: an active particle's factor is NaN or infinite (a model bug or bad hydraulics).
        """
        if np.shape(factor) != (self.n,):
            raise ValueError(f"behave must return a length-{self.n} array or None, got shape {np.shape(factor)}")
        act = self._status == ACTIVE
        bad = act & ~np.isfinite(np.asarray(factor, dtype=np.float64))
        if bad.any():
            i = int(np.nonzero(bad)[0][0])
            raise RuntimeError(f"non-finite velocity factor for particle {i} in reach {int(self._reach[i])}")

    # ---- model hooks (no-ops in the base; the passive model draws nothing here) ----
    def on_release(self, idx: IntArray, h: Hydraulics) -> None:
        """Hook: initialize model state for the particles released this step.

        Called from ``_release`` after the particles are activated, only when ``idx`` is non-empty.

        Args:
            idx: indices of the particles released this step.
            h: the step's hydraulics dict (export names).
        """

    def behave(self, h: Hydraulics, tau: FloatArray, t: float, dt: float) -> FloatArray | None:  # noqa: ARG002
        """Hook: update model state before advection and return a per-particle velocity factor.

        Called after release and before advection. A model may change its declared state, ``mass``
        and ``status`` (through ``terminate``). The returned array multiplies the reach velocity in
        this step's advection, including the time carry across a hop; ``None`` means 1 everywhere.

        Args:
            h: the step's hydraulics dict (export names).
            tau: per-particle time budget for this step (s).
            t: solver clock at the start of the step (s).
            dt: step size (s).

        Returns:
            A length-``n`` array of velocity factors, or None.
        """
        return None

    def diagnostics(self, h: Hydraulics) -> list[str]:  # noqa: ARG002
        """Hook: lines for the startup report about this model on one hydraulics slice (none in the base)."""
        return []

    def _release(self, t: float, dt: float, h: Hydraulics) -> FloatArray:
        """Activate particles due in (t, t + dt] (and any still pending at t); return per-particle time budgets."""
        tau = np.full(self.n, dt)
        new = (self._status == UNRELEASED) & (self.release_time <= t + dt)
        if new.any():
            self._status[new] = ACTIVE
            self._reach[new] = self.release_reach[new]
            self._s[new] = self.release_s[new]
            self._prev_reach[new] = -1
            tau[new] = t + dt - np.maximum(self.release_time[new], t)
            self.on_release(np.nonzero(new)[0], h)
        return tau

    def _advect(self, v: FloatArray, tau: FloatArray, t: float, dt: float, factor: FloatArray | None = None) -> None:
        """Move active particles at ``factor * v[reach]`` for ``tau``, carrying leftover time across hops."""
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
            uj = f[j] * v[rj]  # > 0: s only grew past length because the particle moved
            time_left = (self._s[j] - self._length[rj]) / uj
            self._prev_reach[j] = rj
            nxt = self._to_index[rj]
            exiting = nxt < 0
            self._exit_at_outlet(j[exiting], rj[exiting], t + dt - time_left[exiting])
            m = j[~exiting]
            self._reach[m] = nxt[~exiting]
            self._s[m] = f[m] * v[nxt[~exiting]] * time_left[~exiting]
            idx = m
        raise RuntimeError(f"a particle hopped more than max_hops={self.max_hops} reaches in one advection step")

    def _disperse(self, k: FloatArray, tau: FloatArray, t: float, dt: float, flow_out: FloatArray) -> None:
        """Apply one dispersive kick per active particle, carrying overshoot across reach hops.

        Draws one `rng.standard_normal` per active particle and adds `xi * sqrt(2 * k[reach] * tau)`
        to `s`. Downstream overshoot carries the leftover displacement into the next reach (or exits
        at an outlet with `exit_time = t + dt`). Upstream overshoot is handled by `_hop_upstream`:
        back to `prev_reach` when there is history, else into a parent chosen in proportion to the
        parents' `flow_out`, and reflected (`s = -s`) only at a true headwater.

        Args:
            k: per-reach dispersion coefficient (m^2/s).
            tau: per-particle time budget for this step (s).
            t: solver clock at the start of the step (s).
            dt: step size (s).
            flow_out: per-reach outflow (m3/s) for this step, the parent-choice weights.

        Raises:
            RuntimeError: a particle hops more than `max_hops` reaches in one dispersion step.
        """
        idx = np.nonzero(self._status == ACTIVE)[0]
        if idx.size == 0:
            return
        xi = np.asarray(self.rng.standard_normal(idx.size), dtype=np.float64)
        self._s[idx] += xi * np.sqrt(2.0 * k[self._reach[idx]] * tau[idx])
        for _ in range(self.max_hops):
            r = self._reach[idx].astype(np.int64)
            down = self._s[idx] > self._length[r]
            up = self._s[idx] < 0.0
            if not (down.any() or up.any()):
                return
            # downstream: carry the overshoot into the next reach, or exit at the end of the step
            j = idx[down]
            rj = self._reach[j].astype(np.int64)
            over = self._s[j] - self._length[rj]
            self._prev_reach[j] = rj
            nxt = self._to_index[rj]
            exiting = nxt < 0
            self._exit_at_outlet(j[exiting], rj[exiting], t + dt)
            m = j[~exiting]
            self._reach[m] = nxt[~exiting]
            self._s[m] = over[~exiting]
            # upstream: back the way the particle came, else into a parent, else reflect
            u = idx[up]
            self._hop_upstream(u, flow_out)
            idx = np.concatenate([m, u])
        raise RuntimeError(f"a particle hopped more than max_hops={self.max_hops} reaches in one dispersion step")

    def _hop_upstream(self, u: IntArray, flow_out: FloatArray) -> None:
        """Move particles whose dispersive kick took ``s`` below 0 one reach upstream (hybrid rule).

        A particle with recorded history goes back to `prev_reach` (vectorized, history cleared).
        One with no history goes to a parent of its reach: the only parent, or one drawn from
        `self.rng` with probability proportional to the parents' `flow_out` for this step (uniform
        when every parent's flow is 0). A particle in a true headwater, which has no parent, is
        reflected off the top of its reach (`s = -s`). The parent branch runs as a Python loop over
        the affected particles, which are rare in any one step.

        Args:
            u: indices of the particles with ``s < 0``.
            flow_out: per-reach outflow (m3/s) for this step, the parent-choice weights.
        """
        if u.size == 0:
            return
        pr = self._prev_reach[u].astype(np.int64)
        has = pr >= 0
        a = u[has]
        self._s[a] = self._length[pr[has]] + self._s[a]
        self._reach[a] = pr[has]
        self._prev_reach[a] = -1
        ptr, pidx = self.network.parents_csr()
        for i in u[~has]:
            r = int(self._reach[i])
            par = pidx[ptr[r] : ptr[r + 1]]
            if par.size == 0:  # a true headwater: reflect off the top of the reach
                self._s[i] = -self._s[i]
                continue
            if par.size == 1:
                p = int(par[0])
            else:
                w = np.asarray(flow_out[par], dtype=np.float64)
                w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
                total = float(w.sum())
                p = int(self.rng.choice(par, p=w / total if total > 0.0 else None))
            self._s[i] = self._length[p] + self._s[i]
            self._reach[i] = p
            self._prev_reach[i] = -1
