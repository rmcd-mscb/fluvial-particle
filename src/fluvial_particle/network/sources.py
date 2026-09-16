"""Mass-loading sources and their expansion into a per-particle release schedule."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .config import NetworkConfig, parse_datetime
from .dispersion import dispersion_coefficient
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

    def __post_init__(self) -> None:
        """Check the five arrays describe the same particles.

        Raises:
            ValueError: an array's size differs from ``release_reach.size`` (the message names it),
                which would otherwise surface as a solver crash or a silently truncated run.
        """
        n = int(np.asarray(self.release_reach).size)
        for f in dataclasses.fields(self):
            size = int(np.asarray(getattr(self, f.name)).size)
            if size != n:
                raise ValueError(f"ParticleSchedule.{f.name} has {size} entries, expected {n} (release_reach.size)")

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
            np.array([reach], dtype=np.int32),
            np.array([s], dtype=np.float64),
            np.array([time], dtype=np.float64),
            np.array([mass], dtype=np.float64),
            np.array([source_index], dtype=np.int32),
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


def _curve(
    i: int, row: Mapping[str, Any], key: str, start_time: np.datetime64, total: float
) -> tuple[FloatArray, FloatArray]:
    """Breakpoints (t, value) of a constant-with-window or tabulated curve, restricted to [0, total].

    A tabulated curve has no defined rate outside its own [first, last] time: that region carries
    no mass and is simply omitted (not linearly ramped down to a synthetic zero at 0 or ``total``).
    """
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
            warnings.warn(
                f"sources[{i}] curve extends outside the run window and is truncated", UserWarning, stacklevel=4
            )
        lo, hi = max(float(tp[0]), 0.0), min(float(tp[-1]), total)
        if hi <= lo:
            # The curve's own domain misses the run window entirely: no mass, no breakpoints to add.
            return np.array([0.0, total]), np.array([0.0, 0.0])
        inner = tp[(tp > lo) & (tp < hi)]
        t = np.unique(np.concatenate([[lo, hi], inner]))
        v = np.interp(t, tp, vp)
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
            times = _release_times(row, count, mass_total, cum, t, v, rng)
        parts.append(
            ParticleSchedule(
                np.full(count, idx, dtype=np.int32),
                np.full(count, s, dtype=np.float64),
                times.astype(np.float64),
                masses,
                np.full(count, i, dtype=np.int32),
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
    if particle_mass is None:
        raise ValueError(f"sources[{i}] needs particles, or a global particle_mass")
    if poisson:
        count = max(1, int(rng.poisson(mass_total / particle_mass)))
        return count, np.full(count, mass_total / count)
    count = max(1, round(mass_total / particle_mass))
    masses = np.full(count, particle_mass)
    masses[-1] = mass_total - particle_mass * (count - 1)
    return count, masses


def _invert_cumulative(q: FloatArray, t: FloatArray, v: FloatArray, cum: FloatArray) -> FloatArray:
    """Exactly invert cumulative-mass quantiles ``q`` to release times.

    The rate varies linearly between adjacent (t, v) breakpoints, so cumulative mass is quadratic
    within each segment; segments carrying no mass (a zero-rate span, or a zero-width breakpoint
    pair) are skipped rather than approximated by linearly interpolating the coarse cumulative
    array, which would smear mass across segments where the rate is not actually constant.
    """
    dt = np.diff(t)
    dv = np.diff(v)
    dm = np.diff(cum)
    valid = dm > 0.0
    t0 = t[:-1][valid]
    v0 = v[:-1][valid]
    a = np.divide(dv[valid], dt[valid], out=np.zeros_like(dt[valid]), where=dt[valid] > 0.0)
    bounds = np.concatenate([[0.0], np.cumsum(dm[valid])])
    idx = np.clip(np.searchsorted(bounds, q, side="right") - 1, 0, t0.size - 1)
    local_q = q - bounds[idx]
    a_i, v0_i = a[idx], v0[idx]
    lin = a_i == 0.0
    tau = np.empty_like(q)
    tau[lin] = local_q[lin] / v0_i[lin]
    disc = np.maximum(v0_i[~lin] ** 2 + 2.0 * a_i[~lin] * local_q[~lin], 0.0)
    tau[~lin] = (-v0_i[~lin] + np.sqrt(disc)) / a_i[~lin]
    return t0[idx] + tau


def _release_times(
    row: Mapping[str, Any],
    count: int,
    mass_total: float,
    cum: FloatArray,
    t: FloatArray,
    v: FloatArray,
    rng: np.random.RandomState,
) -> FloatArray:
    spacing = row.get("spacing", "even")
    if spacing == "poisson":
        q = np.sort(rng.uniform(0.0, mass_total, count))
    elif spacing == "even":
        q = (np.arange(count) + 0.5) / count * mass_total
    else:
        raise ValueError(f"spacing must be 'even' or 'poisson', got {spacing!r}")
    return _invert_cumulative(q, t, v, cum)


def _flow_at(
    provider: HydraulicsProvider, start_time: np.datetime64, idx: int, frac: float, t: FloatArray
) -> FloatArray:
    """Flow at fraction ``frac`` along reach ``idx`` at seconds ``t`` (requests in increasing time)."""
    q = np.empty(t.size)
    for j, tj in enumerate(t):
        h = provider.hydraulics(start_time + np.timedelta64(round(tj * 1e9), "ns"))
        q[j] = h["flow_in"][idx] + (h["flow_out"][idx] - h["flow_in"][idx]) * frac
    return q


def _concentration_rate(
    i: int,
    row: Mapping[str, Any],
    provider: HydraulicsProvider,
    start_time: np.datetime64,
    idx: int,
    frac: float,
    total: float,
) -> tuple[FloatArray, FloatArray]:
    """Mass rate C(t) * Q(t) at the union of the curve's breakpoints and the provider's timestamps in between."""
    t_c, c = _curve(i, row, "value", start_time, total)
    ts = (provider.times - start_time) / np.timedelta64(1, "s")
    inner = ts[(ts > t_c[0]) & (ts < t_c[-1])]
    t_all = np.unique(np.concatenate([t_c, inner]))
    c_all = np.interp(t_all, t_c, c, left=0.0, right=0.0)
    return t_all, c_all * _flow_at(provider, start_time, idx, frac, t_all)


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

    Args:
        config: the network run's configuration, for its sources and dispersion settings.
        provider: hydraulics, for velocity and dispersion inputs and the time axis.
        target_per_bin: desired particle count in a peak bin.
        bin_length: bin length (m) used for the estimate.
        reference_travel_time: travel time (s) at which the slug's spread is evaluated.

    Returns:
        DataFrame with columns source, form, reach_id, total_mass, particle_mass, particles.

    Raises:
        ValueError: a continuous source's release reach has a mean velocity of 0 over its window, so
            there is no travel distance to spread its particles over.
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
            h = provider.hydraulics(start + np.timedelta64(round(tj * 1e9), "ns"))
            vel.append(float(h["velocity"][idx]))
            kk.append(
                float(
                    dispersion_coefficient(
                        h, disp.model, scale=disp.scale, cap=disp.cap, value=disp.value, background=disp.background
                    )[idx]
                )
            )
        v_mean = float(np.mean(vel))
        k_mean = float(np.mean(kk))
        if form != "slug" and v_mean <= 0.0:
            raise ValueError(
                f"sources[{i}] on reach {row['reach_id']} has mean velocity {v_mean:g} m/s over its window; "
                "a continuous source needs a positive velocity to size its particles"
            )
        if form == "slug":
            sigma = np.sqrt(2.0 * k_mean * reference_travel_time)
            n = max(1, int(np.ceil(target_per_bin * max(sigma * np.sqrt(2.0 * np.pi) / bin_length, 1.0))))
            m = mass_total / n
        else:
            rate_mean = mass_total / max(t1 - t0, 1e-12)
            m = rate_mean * bin_length / (v_mean * target_per_bin)
            n = max(1, int(np.ceil(mass_total / m)))
        rows.append({
            "source": i,
            "form": form,
            "reach_id": int(row["reach_id"]),
            "total_mass": mass_total,
            "particle_mass": m,
            "particles": n,
        })
    return pd.DataFrame(rows, columns=["source", "form", "reach_id", "total_mass", "particle_mass", "particles"])
