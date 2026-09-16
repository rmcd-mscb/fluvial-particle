"""Vertical profiles for the network drift model: velocity factor, mixing, Taylor's shear coefficient.

Everything here is a pure function of the ``VerticalDispersionConfig`` and the reach hydraulics; no
solver state. The convention is ``zeta = z / h`` with 0 at the bed and 1 at the surface, and the
walk lives on the truncated domain ``[zeta_min, 1 - zeta_min]``. The **same** floored, renormalized
velocity factor and the **same** domain are used by the walk and by the shear-dispersion quadrature,
so the correction to the longitudinal K removes exactly what the resolved vertical motion generates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid


if TYPE_CHECKING:
    from .config import VerticalDispersionConfig

FloatArray = npt.NDArray[np.float64]

# The quadrature grid never touches zeta = 0 (ln 0) or 1 (parabolic Kz = 0 in 1 / Kz).
_EPS = 1e-9
RATIO_MAX = 2.0  # ustar / v above which the tables are held at their last value (near-still reaches)


def _domain(zeta_min: float, n: int) -> FloatArray:
    """The walk's domain ``[zeta_min, 1 - zeta_min]`` as a uniform grid, kept away from 0 and 1."""
    eps = max(float(zeta_min), _EPS)
    return np.linspace(eps, 1.0 - eps, n)


def _log_deviation(zeta: FloatArray, kappa: float) -> FloatArray:
    """``g(zeta) = (1 + ln zeta) / kappa``: the log-law deviation from the depth mean in units of ustar."""
    return (1.0 + np.log(zeta)) / kappa


def _mean(y: FloatArray, z: FloatArray) -> float:
    """Mean of ``y`` over the grid ``z`` by the trapezoid rule."""
    return float(np.trapezoid(y, z) / (z[-1] - z[0]))


def _floored_mean(ratio: FloatArray, kappa: float, lo: float, hi: float) -> FloatArray:
    """Exact domain mean of the floored log-law factor ``max(1 + ratio g, 0)`` over ``[lo, hi]``.

    The factor is positive above ``zeta0 = exp(-1 - kappa / ratio)`` and its integral from
    ``a = max(lo, zeta0)`` to ``hi`` is ``(1 + ratio / kappa) (hi - a) + (ratio / kappa) [zeta ln zeta - zeta]``.
    Closed form, so the walk's factor has unit mean to rounding, not to a table's interpolation.
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    out = np.ones_like(ratio)
    pos = ratio > 0.0
    r = ratio[pos]
    with np.errstate(divide="ignore", over="ignore"):
        a = np.maximum(lo, np.exp(-1.0 - kappa / r))
    prim_hi = hi * np.log(hi) - hi
    prim_a = a * np.log(a) - a
    out[pos] = ((1.0 + r / kappa) * (hi - a) + (r / kappa) * (prim_hi - prim_a)) / (hi - lo)
    return out


def _deviation(z: FloatArray, kappa: float, ratio: float | None) -> FloatArray:
    """The velocity deviation in units of ustar on the domain grid, with zero domain mean.

    With ``ratio = ustar / v`` given, this is ``(f - 1) / ratio`` for the floored, renormalized factor
    ``f = max(1 + ratio g, 0) / mean(max(1 + ratio g, 0))`` that the walk applies; with ``ratio``
    None it is the unfloored ``g - mean(g)`` (the textbook Elder case, the small-ratio limit).
    """
    g = _log_deviation(z, kappa)
    if ratio is None or ratio <= 0.0:
        return g - _mean(g, z)
    raw = np.maximum(1.0 + ratio * g, 0.0)
    f = raw / _floored_mean(np.array([ratio]), kappa, float(z[0]), float(z[-1]))[0]
    return np.asarray((f - 1.0) / ratio, dtype=np.float64)


def _taylor(z: FloatArray, gt: FloatArray, q: FloatArray) -> float:
    """Taylor's dimensionless shear-dispersion integral on the domain ``z``.

    ``K_shear = c ustar h`` with ``c = -(1 / W) int gt * G2``, ``G1 = int gt``, ``G2 = int G1 / q``,
    ``W`` the domain width, ``gt`` the zero-mean velocity deviation in units of ustar and
    ``q = Kz / (ustar h)``. The ``1 / W`` is the cross-sectional average over the walk's domain.
    """
    g1 = cumulative_trapezoid(gt, z, initial=0.0)
    g2 = cumulative_trapezoid(g1 / q, z, initial=0.0)
    return float(-np.trapezoid(gt * g2, z) / (z[-1] - z[0]))


def shear_dispersion_coefficient(
    cfg: VerticalDispersionConfig,
    zeta_min: float,
    ratio: float | None = None,
    *,
    ustar_depth: float | None = None,
    n: int = 200001,
) -> float:
    """Taylor's shear-dispersion coefficient ``c`` (``K_shear = c ustar h``) by quadrature on the walk's domain.

    On the full column with the parabolic profile and the log law this is Elder's ``0.404 / kappa^3``
    (5.86 at ``kappa = 0.41``); truncating at ``zeta_min`` removes the slow near-bed layer (5.65 at
    0.001, 4.62 at 0.01); the constant profile at ``beta = 0.067`` gives 6.58.

    Args:
        cfg: the vertical profiles.
        zeta_min: the walk's domain is ``[zeta_min, 1 - zeta_min]``.
        ratio: ``ustar / v``; when given the velocity factor is floored at 0 and renormalized to
            unit mean exactly as the walk applies it, else the unfloored log law is used.
        ustar_depth: ``ustar * h`` (m^2/s) of the reach, needed only when ``Kz`` has a part that does
            not scale with it (the ``"value"`` profile or a non-zero ``background``).
        n: quadrature points.

    Returns:
        ``c``; 0 for the uniform velocity profile, and 0 when ``Kz`` is 0 (no vertical mixing means
        no Taylor regime and nothing for the walk to generate as dispersion within a step). A
        ``"value"`` profile or a non-zero background without ``ustar_depth`` raises ValueError.
    """
    if cfg.velocity_profile == "uniform":
        return 0.0
    z = _domain(zeta_min, n)
    q = _dimensionless_kz(cfg, z, ustar_depth)
    if not np.all(q > 0.0):
        return 0.0
    gt = _deviation(z, cfg.kappa, ratio)
    return _taylor(z, gt, q)


def _dimensionless_kz(cfg: VerticalDispersionConfig, z: FloatArray, ustar_depth: float | None) -> FloatArray:
    """``q = Kz / (ustar h)`` on the grid.

    Raises:
        ValueError: the profile needs ``ustar_depth`` and none was given.
    """
    if cfg.profile == "parabolic":
        q = cfg.scale * cfg.kappa * z * (1.0 - z)
    elif cfg.profile == "constant":
        q = np.full_like(z, cfg.scale * cfg.beta)
    else:
        if ustar_depth is None:
            raise ValueError("the 'value' profile needs ustar_depth (ustar * h) to form Kz / (ustar h)")
        assert cfg.value is not None
        q = np.full_like(z, cfg.scale * cfg.value / ustar_depth)
    if cfg.background > 0.0:
        if ustar_depth is None:
            raise ValueError("a non-zero vertical background needs ustar_depth (ustar * h) to form Kz / (ustar h)")
        q += cfg.background / ustar_depth
    return q


class VerticalProfiles:
    """Velocity factor and vertical mixing profiles for one ``VerticalDispersionConfig`` and ``zeta_min``.

    Taylor's coefficient ``c(ratio)`` is tabulated at construction on a grid of ``ratio = ustar / v``
    in ``[0, RATIO_MAX]`` and interpolated per reach and step, so the per-step cost is nil; the
    normalization of the floored log-law factor is exact (closed form). The ``"value"`` profile and
    a non-zero ``background`` make ``c`` depend on ``ustar * h`` as well; for those
    ``shear_coefficient`` runs the quadrature per reach on a coarser grid (``n_reach`` x
    ``n_reach_quad`` points per step).

    Args:
        cfg: the vertical profiles.
        zeta_min: the walk's domain is ``[zeta_min, 1 - zeta_min]``.
        n_ratio: table points in ``ratio``.
        n_quad: quadrature points for the tables.
        n_reach_quad: quadrature points for the per-reach path.
    """

    def __init__(
        self,
        cfg: VerticalDispersionConfig,
        zeta_min: float,
        *,
        n_ratio: int = 65,
        n_quad: int = 20001,
        n_reach_quad: int = 2001,
    ) -> None:
        """Build the normalization and shear-coefficient tables."""
        self.cfg = cfg
        self.zeta_min = float(zeta_min)
        self.n_reach_quad = int(n_reach_quad)
        self._log = cfg.velocity_profile == "log"
        self._per_reach = cfg.profile == "value" or cfg.background > 0.0
        self._ratios = np.linspace(0.0, RATIO_MAX, n_ratio)
        z = _domain(zeta_min, n_quad)
        self._lo, self._hi = float(z[0]), float(z[-1])
        if self._log and not self._per_reach:
            q = _dimensionless_kz(cfg, z, None)
            self._shear = np.array([_taylor(z, _deviation(z, cfg.kappa, r), q) for r in self._ratios])
        else:
            self._shear = np.zeros(n_ratio)

    # ---- velocity ---------------------------------------------------------------
    def velocity_factor(self, zeta: FloatArray, ustar: FloatArray, v: FloatArray) -> FloatArray:
        """``u(zeta) / v`` per particle: the floored, renormalized log law, or 1 for the uniform profile.

        Args:
            zeta: relative elevations.
            ustar: shear velocity of each particle's reach (m/s).
            v: velocity of each particle's reach (m/s); where it is 0 the factor is 1.

        Returns:
            The velocity factor, non-negative, with unit mean over the domain.
        """
        zeta = np.asarray(zeta, dtype=np.float64)
        if not self._log:
            return np.ones_like(zeta)
        v = np.asarray(v, dtype=np.float64)
        ustar = np.asarray(ustar, dtype=np.float64)
        ratio = np.zeros_like(zeta)
        moving = v > 0.0
        ratio[moving] = ustar[moving] / v[moving]
        raw = np.maximum(1.0 + ratio * _log_deviation(np.maximum(zeta, _EPS), self.cfg.kappa), 0.0)
        return np.asarray(raw / _floored_mean(ratio, self.cfg.kappa, self._lo, self._hi), dtype=np.float64)

    # ---- mixing -----------------------------------------------------------------
    def kz(self, zeta: FloatArray, ustar: FloatArray, h: FloatArray) -> FloatArray:
        """Vertical mixing coefficient ``Kz(zeta)`` (m^2/s) per particle."""
        cfg = self.cfg
        zeta = np.asarray(zeta, dtype=np.float64)
        if cfg.profile == "parabolic":
            base = cfg.scale * cfg.kappa * np.asarray(ustar) * np.asarray(h) * zeta * (1.0 - zeta)
        elif cfg.profile == "constant":
            base = cfg.scale * cfg.beta * np.asarray(ustar) * np.asarray(h) * np.ones_like(zeta)
        else:
            assert cfg.value is not None
            base = np.full_like(zeta, cfg.scale * cfg.value)
        return np.asarray(base + cfg.background, dtype=np.float64)

    def dkz_dz(self, zeta: FloatArray, ustar: FloatArray, h: FloatArray) -> FloatArray:  # noqa: ARG002
        """``dKz/dz`` (m/s) per particle, the gradient drift of the Ito walk; analytic for every profile."""
        cfg = self.cfg
        zeta = np.asarray(zeta, dtype=np.float64)
        if cfg.profile == "parabolic":
            return np.asarray(cfg.scale * cfg.kappa * np.asarray(ustar) * (1.0 - 2.0 * zeta), dtype=np.float64)
        return np.zeros_like(zeta)

    def kz_max(self, ustar: FloatArray, h: FloatArray) -> FloatArray:
        """The profile's maximum ``Kz`` (m^2/s) per reach, for the sub-step rule."""
        cfg = self.cfg
        ustar = np.asarray(ustar, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        if cfg.profile == "parabolic":
            base = cfg.scale * cfg.kappa * ustar * h / 4.0
        elif cfg.profile == "constant":
            base = cfg.scale * cfg.beta * ustar * h
        else:
            assert cfg.value is not None
            base = np.full_like(ustar, cfg.scale * cfg.value)
        return np.asarray(base + cfg.background, dtype=np.float64)

    # ---- shear dispersion -------------------------------------------------------
    def shear_coefficient(self, ustar: FloatArray, v: FloatArray, h: FloatArray | None = None) -> FloatArray:
        """Taylor's coefficient ``c`` per reach (``K_shear = c ustar h``); 0 where ``v`` or ``ustar`` is 0.

        Args:
            ustar: shear velocity per reach (m/s).
            v: velocity per reach (m/s).
            h: depth per reach (m); required for the ``"value"`` profile or a non-zero ``background``.

        Returns:
            ``c`` per reach.

        Raises:
            ValueError: the profile needs ``h`` and none was given.
        """
        ustar = np.asarray(ustar, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        c = np.zeros_like(ustar)
        if not self._log:
            return c
        ok = (v > 0.0) & (ustar > 0.0)
        if not ok.any():
            return c
        ratio = ustar[ok] / v[ok]
        if not self._per_reach:
            c[ok] = np.interp(ratio, self._ratios, self._shear)
            return c
        if h is None:
            raise ValueError(f"the {self.cfg.profile!r} profile or a vertical background needs depth per reach")
        uh = ustar[ok] * np.asarray(h, dtype=np.float64)[ok]
        c[ok] = [
            shear_dispersion_coefficient(self.cfg, self.zeta_min, float(r), ustar_depth=float(s), n=self.n_reach_quad)
            for r, s in zip(ratio, uh, strict=True)
        ]
        return c


def substep_counts(dt: float, kz_max: FloatArray, h: FloatArray, *, c: float = 0.1) -> npt.NDArray[np.int64]:
    """Uncapped sub-steps per reach, ``ceil(dt / (c h^2 / Kz_max))``; 1 where there is no depth or no mixing."""
    kz_max = np.asarray(kz_max, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    ok = (h > 0.0) & (kz_max > 0.0)
    n = np.ones(kz_max.shape, dtype=np.int64)
    n[ok] = np.ceil(float(dt) * kz_max[ok] / (c * h[ok] ** 2) - 1e-9).astype(np.int64)
    return np.maximum(n, 1)


def substep_count(dt: float, kz_max: FloatArray, h: FloatArray, *, c: float = 0.1, max_substeps: int = 500) -> int:
    """Sub-steps for the vertical walk: ``ceil(dt / (c h^2 / Kz_max))`` over the reaches, capped.

    One count for all active particles (the maximum over their reaches) keeps the walk vectorized.
    Reaches with no depth or no mixing do not constrain the count; the result is at least 1.

    Args:
        dt: the solver step (s).
        kz_max: the profile's maximum ``Kz`` per reach (m^2/s).
        h: depth per reach (m).
        c: the fraction of the column mixing time ``h^2 / Kz_max`` one sub-step may span.
        max_substeps: cap on the count.

    Returns:
        The sub-step count.
    """
    counts = substep_counts(dt, kz_max, h, c=c)
    n = int(counts.max()) if counts.size else 1
    return int(min(max(n, 1), max_substeps))


def deposition_probability(k_d: FloatArray, dt_sub: float | FloatArray, kz_bed: FloatArray) -> FloatArray:
    """Per-contact deposition probability reproducing a Robin bed condition with deposition velocity ``k_d``.

    ``p = k_d sqrt(pi dt_sub / Kz(zeta_min))`` (Erban and Chapman 2007), clipped to ``[0, 1]``: the
    probability that a particle whose sub-step crosses the bed deposits rather than reflects, chosen
    so the mean flux into the bed is ``k_d`` times the near-bed concentration for any sub-step length.
    Where ``Kz`` at the bed is 0 nothing reaches it by diffusion and the probability is 0.

    Args:
        k_d: deposition velocity per particle (m/s), after any critical-shear suppression.
        dt_sub: sub-step length (s), scalar or per particle.
        kz_bed: ``Kz`` at ``zeta_min`` per particle (m^2/s).

    Returns:
        The probability per particle.
    """
    k_d = np.asarray(k_d, dtype=np.float64)
    kz_bed = np.asarray(kz_bed, dtype=np.float64)
    dts = np.broadcast_to(np.asarray(dt_sub, dtype=np.float64), k_d.shape)
    p = np.zeros_like(k_d)
    ok = kz_bed > 0.0
    p[ok] = k_d[ok] * np.sqrt(np.pi * dts[ok] / kz_bed[ok])
    return np.clip(p, 0.0, 1.0)
