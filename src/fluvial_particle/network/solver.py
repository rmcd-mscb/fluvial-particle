"""Vectorized 1D network particle solver: exact advection with time carry, then one dispersive kick."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .config import DispersionConfig
from .dispersion import dispersion_coefficient
from .network import Network
from .provider import HydraulicsProvider
from .sources import ParticleSchedule


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
TERMINAL_STATUSES = (EXITED, SETTLED, REMOVED)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _readonly(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """A non-writable view of ``arr`` that still tracks the backing array's contents."""
    view = arr.view()
    view.setflags(write=False)
    return view


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
        """
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

    # ---- particle state (read-only views of the solver's own arrays) --------
    @property
    def reach(self) -> npt.NDArray[np.int32]:
        """Reach index of each particle, -1 when unreleased or exited (settled particles keep theirs)."""
        return self._views["reach"]

    @property
    def s(self) -> npt.NDArray[Any]:
        """Distance from the upstream end of ``reach`` (m), NaN when unreleased or exited (settled keep theirs)."""
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

    def terminate(self, idx: IntArray, status: int, t_end: float) -> None:
        """Give particles ``idx`` a terminal status at time ``t_end``; they keep their reach and ``s``.

        Used by behavioral models (a settled particle has a position on the bed). The base solver's
        exits keep their own inline bookkeeping in ``_advect`` and ``_disperse``.

        Args:
            idx: particle indices.
            status: one of ``TERMINAL_STATUSES``.
            t_end: solver time (s) at which the particles reached the status.
        """
        if idx.size == 0:
            return
        self._status[idx] = status
        self._exit_time[idx] = t_end
        self._exit_reach[idx] = self._reach[idx]

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
        """Advance the clock by dt: release, advect with time carry, disperse with displacement carry."""
        t = self.time
        dt = self.dt
        tau = self._release(t, dt)
        h = self.provider.hydraulics(self.midpoint_time())
        v = np.asarray(h["velocity"], dtype=np.float64)
        d = self.dispersion
        k = dispersion_coefficient(h, d.model, scale=d.scale, cap=d.cap, value=d.value)
        self._advect(v, tau, t, dt)
        self._disperse(k, tau, t, dt, np.asarray(h["flow_out"], dtype=np.float64))
        self.time = t + dt

    def _release(self, t: float, dt: float) -> FloatArray:
        """Activate particles due in (t, t + dt] (and any still pending at t); return per-particle time budgets."""
        tau = np.full(self.n, dt)
        new = (self._status == UNRELEASED) & (self.release_time <= t + dt)
        if new.any():
            self._status[new] = ACTIVE
            self._reach[new] = self.release_reach[new]
            self._s[new] = self.release_s[new]
            self._prev_reach[new] = -1
            tau[new] = t + dt - np.maximum(self.release_time[new], t)
        return tau

    def _advect(self, v: FloatArray, tau: FloatArray, t: float, dt: float) -> None:
        idx = np.nonzero(self._status == ACTIVE)[0]
        if idx.size == 0:
            return
        self._s[idx] += v[self._reach[idx]] * tau[idx]
        for _ in range(self.max_hops):
            over = self._s[idx] > self._length[self._reach[idx]]
            if not over.any():
                return
            j = idx[over]
            rj = self._reach[j].astype(np.int64)
            time_left = (self._s[j] - self._length[rj]) / v[rj]
            self._prev_reach[j] = rj
            nxt = self._to_index[rj]
            exiting = nxt < 0
            e = j[exiting]
            self._status[e] = EXITED
            self._exit_time[e] = t + dt - time_left[exiting]
            self._exit_reach[e] = rj[exiting]
            self._reach[e] = -1
            self._s[e] = np.nan
            m = j[~exiting]
            self._reach[m] = nxt[~exiting]
            self._s[m] = v[nxt[~exiting]] * time_left[~exiting]
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
            e = j[exiting]
            self._status[e] = EXITED
            self._exit_time[e] = t + dt
            self._exit_reach[e] = rj[exiting]
            self._reach[e] = -1
            self._s[e] = np.nan
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
