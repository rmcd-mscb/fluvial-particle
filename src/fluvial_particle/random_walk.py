"""Numerical pieces of the random walk shared by the 2D/3D and network solvers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def reflect_interval(x: npt.NDArray[np.float64], lo: npt.ArrayLike, hi: npt.ArrayLike) -> npt.NDArray[np.float64]:
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


def reciprocal_or_zero(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """``1 / x`` where ``x`` is positive, 0 elsewhere: a reach without depth has nothing to divide by.

    Args:
        x: values to invert (a depth, say).

    Returns:
        The reciprocal, a new array, with 0 wherever ``x`` is not positive (division is never
        attempted there, so no warning is raised).
    """
    a = np.asarray(x, dtype=np.float64)
    return np.divide(1.0, a, out=np.zeros_like(a), where=a > 0.0)
