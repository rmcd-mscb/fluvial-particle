"""Vectorized 1D network particle solver: exact advection with time carry, then one dispersive kick."""

from __future__ import annotations

from typing import Any, cast

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
        return self.start_time + np.timedelta64(round((self.time + 0.5 * self.dt) * 1e9), "ns")

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

    def _disperse(self, _k: FloatArray, _tau: FloatArray, _t: float, _dt: float) -> None:
        """Replaced in the dispersion task."""
        return
