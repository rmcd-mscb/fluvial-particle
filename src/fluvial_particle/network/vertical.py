"""Vertical profiles for the network drift model: velocity factor, mixing kernels, Taylor's shear coefficient.

Everything here is a pure function of the ``VerticalDispersionConfig`` and the reach hydraulics; no
solver state. The convention is ``zeta = z / h`` with 0 at the bed and 1 at the surface. The walk
lives on the full column ``[0, 1]``; ``zeta_min`` is the clip below which the log-law velocity
factor is held at its ``zeta_min`` value (and above ``1 - zeta_min`` likewise) and the thickness of
the bed contact layer in which deposition acts. The **same** clipped, floored, renormalized velocity
factor is used by the walk and by the shear-dispersion quadrature, so the correction to the
longitudinal K removes exactly what the resolved vertical motion generates.

Mixing kernels (both leave a uniform column exactly uniform at any step, so the well-mixed condition
holds by construction rather than to some order in the step):

- parabolic ``Kz = kappa ustar h zeta (1 - zeta)``: the Ito walk ``d zeta = a (1 - 2 zeta) dt +
  sqrt(2 a zeta (1 - zeta)) dW`` with ``a = kappa ustar / h`` is a Jacobi (Wright-Fisher) diffusion,
  and with ``zeta = (1 - cos theta) / 2`` it is the polar angle of Brownian motion on the unit sphere
  with generator ``a`` times the spherical Laplacian (Karlin and Taylor 1981, ch. 15). One sub-step
  is therefore a random rotation: a von Mises-Fisher step about the current direction (Fisher 1953),
  whose cosine has a closed-form inverse CDF, with the concentration calibrated so the first
  spherical mode decays as the heat kernel does, ``E[cos psi] = exp(-2 a dt)``. Isotropy makes the
  uniform measure invariant for every step; higher modes are approximated to first order in ``a dt``.
- constant or fixed ``Kz``: a Gaussian displacement mirror-reflected at 0 and 1, exact for a
  constant coefficient.

The Euler-Ito walk with the gradient drift and a reflecting wall at ``zeta_min`` (Visser 1997; Ross
and Sharples 2004) was measured and rejected: with ``Kz`` vanishing linearly at the wall the drift
step must be far smaller than ``zeta_min``, about a hundredth of ``1 / |Kz''|``, to keep the column
uniform, which is out of reach for a network run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy.integrate import cumulative_trapezoid

from ..random_walk import reflect_interval


if TYPE_CHECKING:
    from .config import VerticalDispersionConfig

FloatArray = npt.NDArray[np.float64]

# The quadrature grid never touches zeta = 0 (ln 0) or 1 (parabolic Kz = 0 in 1 / Kz).
_EPS = 1e-9
RATIO_MAX = 2.0  # ustar / v above which the shear table is held at its last value (near-still reaches)
SUBSTEP_FRACTION = 0.03  # default fraction of the column mixing time h^2 / Kz_max per sub-step
# von Mises-Fisher calibration table: dimensionless step tau = a dt in [TAU_MIN, TAU_MAX]; below it the
# small-step limit kappa = 1 / (2 tau) holds to 1e-5, above it the step is a fresh uniform direction.
TAU_MIN, TAU_MAX = 1e-5, 12.0


def _grid(n: int) -> FloatArray:
    """The full column ``[0, 1]`` as a uniform quadrature grid, kept away from 0 and 1."""
    return np.linspace(_EPS, 1.0 - _EPS, n)


def _log_deviation(zeta: FloatArray, kappa: float) -> FloatArray:
    """``g(zeta) = (1 + ln zeta) / kappa``: the log-law deviation from the depth mean in units of ustar."""
    return (1.0 + np.log(zeta)) / kappa


def _floored_mean(ratio: FloatArray, kappa: float, lo: float, hi: float) -> FloatArray:
    """Exact full-column mean of the clipped, floored log-law factor ``max(1 + ratio g(clip(zeta)), 0)``.

    The factor is positive above ``zeta0 = exp(-1 - kappa / ratio)``; its integral over ``[a, hi]``
    with ``a = max(lo, zeta0)`` is ``(1 + ratio / kappa) (hi - a) + (ratio / kappa) [zeta ln zeta - zeta]``,
    and the clipped ends contribute ``lo * factor(lo)`` and ``(1 - hi) * factor(hi)``. Closed form, so
    the walk's factor has unit mean to rounding, not to a table's interpolation.
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    out = np.ones_like(ratio)
    pos = ratio > 0.0
    r = ratio[pos]
    with np.errstate(divide="ignore", over="ignore"):
        a = np.maximum(lo, np.exp(-1.0 - kappa / r))
    inner = (1.0 + r / kappa) * (hi - a) + (r / kappa) * ((hi * np.log(hi) - hi) - (a * np.log(a) - a))
    ends = lo * np.maximum(1.0 + r * _log_deviation(np.array(lo), kappa), 0.0)
    ends += (1.0 - hi) * np.maximum(1.0 + r * _log_deviation(np.array(hi), kappa), 0.0)
    out[pos] = inner + ends
    return out


def _deviation(z: FloatArray, kappa: float, ratio: float | None, lo: float, hi: float) -> FloatArray:
    """The velocity deviation in units of ustar on the column grid, with zero column mean.

    With ``ratio = ustar / v`` given, this is ``(f - 1) / ratio`` for the clipped, floored,
    renormalized factor the walk applies; with ``ratio`` None it is the unfloored, clipped
    ``g - mean(g)`` (the textbook Elder case, the small-ratio limit).
    """
    g = _log_deviation(np.clip(z, lo, hi), kappa)
    if ratio is None or ratio <= 0.0:
        return g - float(np.trapezoid(g, z) / (z[-1] - z[0]))
    raw = np.maximum(1.0 + ratio * g, 0.0)
    f = raw / _floored_mean(np.array([ratio]), kappa, lo, hi)[0]
    return np.asarray((f - 1.0) / ratio, dtype=np.float64)


def _taylor(z: FloatArray, gt: FloatArray, q: FloatArray) -> float:
    """Taylor's dimensionless shear-dispersion integral on the column ``z`` (Taylor 1953; Elder 1959).

    ``K_shear = c ustar h`` with ``c = -(1 / W) int gt * G2``, ``G1 = int gt``, ``G2 = int G1 / q``,
    ``W`` the domain width, ``gt`` the zero-mean velocity deviation in units of ustar and
    ``q = Kz / (ustar h)``. The ``1 / W`` is the cross-sectional average.
    """
    g1 = cumulative_trapezoid(gt, z, initial=0.0)
    g2 = cumulative_trapezoid(g1 / q, z, initial=0.0)
    return float(-np.trapezoid(gt * g2, z) / (z[-1] - z[0]))


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


def shear_dispersion_coefficient(
    cfg: VerticalDispersionConfig,
    zeta_min: float,
    ratio: float | None = None,
    *,
    ustar_depth: float | None = None,
    n: int = 200001,
) -> float:
    """Taylor's shear-dispersion coefficient ``c`` (``K_shear = c ustar h``) by quadrature on the column.

    With the parabolic profile and the unclipped log law this is Elder's ``0.404 / kappa^3`` (5.86 at
    ``kappa = 0.41``; Elder 1959); clipping the factor at ``zeta_min`` holds the near-bed velocity at
    its ``zeta_min`` value and lowers it (5.83 at 0.001, 5.58 at 0.01); the constant profile at
    ``beta = 0.067`` gives 6.58.

    Args:
        cfg: the vertical profiles.
        zeta_min: the velocity factor is evaluated at ``clip(zeta, zeta_min, 1 - zeta_min)``.
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
    z = _grid(n)
    q = _dimensionless_kz(cfg, z, ustar_depth)
    if not np.all(q > 0.0):
        return 0.0
    lo = max(float(zeta_min), _EPS)
    gt = _deviation(z, cfg.kappa, ratio, lo, 1.0 - lo)
    return _taylor(z, gt, q)


def _vmf_concentration_table(n: int = 400) -> tuple[FloatArray, FloatArray]:
    """``kappa(tau)`` solving ``coth kappa - 1 / kappa = exp(-2 tau)`` on a log grid of ``tau``."""
    taus = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), n)
    kappas = np.empty(n)
    for i, tau in enumerate(taus):
        target = float(np.exp(-2.0 * tau))
        kappas[i] = optimize.brentq(lambda k, t=target: (1.0 / np.tanh(k) - 1.0 / k) - t, 1e-9, 1e7)
    return taus, kappas


_VMF_TAU, _VMF_KAPPA = _vmf_concentration_table()


def vmf_concentration(tau: FloatArray) -> FloatArray:
    """The von Mises-Fisher concentration giving ``E[cos psi] = exp(-2 tau)`` for the step ``tau = a dt``.

    ``kappa = 1 / (2 tau)`` below ``TAU_MIN`` (the diffusive limit, mean squared angle ``4 tau``); 0
    above ``TAU_MAX`` (a fresh uniform direction); interpolated in log-log between.
    """
    tau = np.asarray(tau, dtype=np.float64)
    out = np.zeros_like(tau)
    small = (tau > 0.0) & (tau < TAU_MIN)
    out[small] = 0.5 / tau[small]
    mid = (tau >= TAU_MIN) & (tau <= TAU_MAX)
    out[mid] = np.exp(np.interp(np.log(tau[mid]), np.log(_VMF_TAU), np.log(_VMF_KAPPA)))
    out[tau <= 0.0] = np.inf
    return out


def vmf_cosine(kappa: FloatArray, u: FloatArray) -> FloatArray:
    """Cosine of the von Mises-Fisher step angle on the sphere by inverse CDF: ``w ~ exp(kappa w)`` on [-1, 1].

    ``w = 1 + ln(1 - u (1 - exp(-2 kappa))) / kappa`` (Ulrich 1984; Wood 1994), written with ``expm1``
    and ``log1p`` so it is exact as ``kappa -> 0`` (uniform ``w``) and ``kappa -> inf`` (``w = 1``).
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    w = np.ones_like(kappa)
    finite = np.isfinite(kappa)
    zero = finite & (kappa <= 1e-12)
    w[zero] = 1.0 - 2.0 * u[zero]
    k = finite & ~zero
    w[k] = 1.0 + np.log1p(-u[k] * (-np.expm1(-2.0 * kappa[k]))) / kappa[k]
    return np.clip(w, -1.0, 1.0)


def sphere_step(zeta: FloatArray, tau: FloatArray, rng: np.random.RandomState) -> FloatArray:
    """One exactly-well-mixed sub-step of the parabolic-profile walk: a random rotation on the sphere.

    ``zeta = (1 - cos theta) / 2``; the new polar cosine is ``cos theta cos psi + sin theta sin psi cos beta``
    with ``psi`` the von Mises-Fisher step angle for ``tau = a dt`` and ``beta`` a uniform azimuth
    (the azimuth of the particle itself is immaterial by symmetry and never stored). Draws two
    uniforms per particle from ``rng``.

    Args:
        zeta: relative elevations in ``[0, 1]``.
        tau: ``a dt = scale kappa ustar dt / h`` per particle (0 leaves the particle where it is).
        rng: random state.

    Returns:
        The new relative elevations, in ``[0, 1]``.
    """
    zeta = np.asarray(zeta, dtype=np.float64)
    cos_t = 1.0 - 2.0 * zeta
    sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    w = vmf_cosine(vmf_concentration(tau), rng.uniform(size=zeta.size))
    beta = rng.uniform(0.0, 2.0 * np.pi, zeta.size)
    cos_new = cos_t * w + sin_t * np.sqrt(np.maximum(1.0 - w**2, 0.0)) * np.cos(beta)
    return np.asarray(np.clip((1.0 - cos_new) / 2.0, 0.0, 1.0), dtype=np.float64)


class VerticalProfiles:
    """Velocity factor, mixing profile and kernel, and shear coefficient for one ``VerticalDispersionConfig``.

    Taylor's coefficient ``c(ratio)`` is tabulated at construction on a grid of ``ratio = ustar / v``
    in ``[0, RATIO_MAX]`` and interpolated per reach and step, so the per-step cost is nil; the
    normalization of the floored log-law factor is exact (closed form). The ``"value"`` profile and
    a non-zero ``background`` make ``c`` depend on ``ustar * h`` as well; for those
    ``shear_coefficient`` runs the quadrature per reach on a coarser grid (``n_reach`` x
    ``n_reach_quad`` points per step).

    Args:
        cfg: the vertical profiles.
        zeta_min: clip of the velocity factor; the walk itself lives on ``[0, 1]``.
        n_ratio: table points in ``ratio``.
        n_quad: quadrature points for the table.
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
        """Build the shear-coefficient table."""
        self.cfg = cfg
        self.zeta_min = float(zeta_min)
        self.n_reach_quad = int(n_reach_quad)
        self._log = cfg.velocity_profile == "log"
        self._per_reach = cfg.profile == "value" or cfg.background > 0.0
        self._lo = max(self.zeta_min, _EPS)
        self._hi = 1.0 - self._lo
        self._ratios = np.linspace(0.0, RATIO_MAX, n_ratio)
        if self._log and not self._per_reach:
            z = _grid(n_quad)
            q = _dimensionless_kz(cfg, z, None)
            self._shear = np.array([
                _taylor(z, _deviation(z, cfg.kappa, r, self._lo, self._hi), q) for r in self._ratios
            ])
        else:
            self._shear = np.zeros(n_ratio)

    # ---- velocity ---------------------------------------------------------------
    def velocity_factor(self, zeta: FloatArray, ustar: FloatArray, v: FloatArray) -> FloatArray:
        """``u(zeta) / v`` per particle: the clipped, floored, renormalized log law, or 1 for the uniform profile.

        Args:
            zeta: relative elevations; evaluated at ``clip(zeta, zeta_min, 1 - zeta_min)``.
            ustar: shear velocity of each particle's reach (m/s).
            v: velocity of each particle's reach (m/s); where it is 0 the factor is 1.

        Returns:
            The velocity factor, non-negative, with unit mean over the column.
        """
        zeta = np.asarray(zeta, dtype=np.float64)
        if not self._log:
            return np.ones_like(zeta)
        v = np.asarray(v, dtype=np.float64)
        ustar = np.asarray(ustar, dtype=np.float64)
        ratio = np.zeros_like(zeta)
        moving = v > 0.0
        ratio[moving] = ustar[moving] / v[moving]
        raw = np.maximum(1.0 + ratio * _log_deviation(np.clip(zeta, self._lo, self._hi), self.cfg.kappa), 0.0)
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

    def mix(
        self, zeta: FloatArray, ustar: FloatArray, h: FloatArray, dt: FloatArray, rng: np.random.RandomState
    ) -> FloatArray:
        """One mixing sub-step of length ``dt`` per particle; the uniform column is exactly invariant.

        Parabolic: the sphere rotation with ``tau = scale kappa ustar dt / h``. Constant or value: a
        Gaussian displacement ``sqrt(2 Kz dt) / h`` mirror-reflected at 0 and 1. A non-zero
        ``background`` on the parabolic profile adds the reflected Gaussian step for its part.
        Particles in a reach without depth are left where they are.

        Args:
            zeta: relative elevations in ``[0, 1]``.
            ustar: shear velocity per particle (m/s).
            h: depth per particle (m).
            dt: sub-step length per particle (s).
            rng: random state.

        Returns:
            The new relative elevations, in ``[0, 1]``.
        """
        cfg = self.cfg
        zeta = np.asarray(zeta, dtype=np.float64)
        ustar = np.asarray(ustar, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        dt = np.asarray(dt, dtype=np.float64)
        wet = h > 0.0
        inv_h = np.where(wet, 1.0 / np.where(wet, h, 1.0), 0.0)
        if cfg.profile == "parabolic":
            zeta = sphere_step(zeta, cfg.scale * cfg.kappa * ustar * dt * inv_h, rng)
            extra = np.full_like(zeta, cfg.background)
        else:
            extra = self.kz(zeta, ustar, h)
        if np.any(extra > 0.0):
            step = np.sqrt(2.0 * extra * dt) * inv_h * rng.standard_normal(zeta.size)
            zeta = reflect_interval(zeta + step, 0.0, 1.0)
        return zeta

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


def substep_counts(
    dt: float, kz_max: FloatArray, h: FloatArray, *, c: float = SUBSTEP_FRACTION
) -> npt.NDArray[np.int64]:
    """Uncapped sub-steps per reach, ``ceil(dt / (c h^2 / Kz_max))``; 1 where there is no depth or no mixing."""
    kz_max = np.asarray(kz_max, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    ok = (h > 0.0) & (kz_max > 0.0)
    n = np.ones(kz_max.shape, dtype=np.int64)
    n[ok] = np.ceil(float(dt) * kz_max[ok] / (c * h[ok] ** 2) - 1e-9).astype(np.int64)
    return np.maximum(n, 1)


def substep_count(
    dt: float, kz_max: FloatArray, h: FloatArray, *, c: float = SUBSTEP_FRACTION, max_substeps: int = 1000
) -> int:
    """Sub-steps for the vertical walk: ``ceil(dt / (c h^2 / Kz_max))`` over the reaches, capped.

    One count for all active particles (the maximum over their reaches) keeps the walk vectorized.
    The kernels keep the column well mixed at any step; ``c`` sets how well the within-step shear
    dispersion is resolved (2 percent at 0.03, 12 percent at 0.1 for the log law with the parabolic
    profile). Reaches with no depth or no mixing do not constrain the count; the result is at least 1.

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


def deposition_probability(
    k_d: FloatArray, dt_sub: float | FloatArray, layer: FloatArray, settling: float = 0.0
) -> FloatArray:
    """Per-contact deposition probability that reproduces a Robin bed condition with deposition velocity ``k_d``.

    A contact is a sub-step whose settling-shifted position, before reflection, lies in the bed
    contact layer ``[0, zeta_min)`` or below it. Per sub-step the particles making contact are those
    within ``layer + w dt_sub`` of the bed (``layer = zeta_min h``, ``w`` the downward velocity), so
    removing each with probability ``p = k_d dt_sub / (layer + w dt_sub)`` gives the Robin flux
    ``k_d C_bed`` while ``p < 1``: the layer rule ``k_d dt_sub / layer`` when contact is by mixing,
    ``k_d / w`` when settling dominates. Clipped to 1 the bed absorbs every particle that reaches it
    (the perfectly absorbing bed: the settling flux plus the mixing supply, the latter growing with
    the sub-step count, which is the bias bound the docs state). The probability is derived, never a
    user parameter. Where the layer has no thickness (a dry reach) it is 0.

    Args:
        k_d: deposition velocity per particle (m/s), after any critical-shear suppression.
        dt_sub: sub-step length (s), scalar or per particle.
        layer: contact-layer thickness ``zeta_min h`` per particle (m).
        settling: vertical velocity (m/s, positive down); only a downward velocity adds contacts.

    Returns:
        The probability per particle.
    """
    k_d = np.asarray(k_d, dtype=np.float64)
    layer = np.asarray(layer, dtype=np.float64)
    dts = np.broadcast_to(np.asarray(dt_sub, dtype=np.float64), k_d.shape)
    reach = layer + max(float(settling), 0.0) * dts
    p = np.zeros_like(k_d)
    ok = reach > 0.0
    p[ok] = k_d[ok] * dts[ok] / reach[ok]
    return np.clip(p, 0.0, 1.0)
