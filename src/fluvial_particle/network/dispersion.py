"""Longitudinal dispersion coefficients for 1D river-network transport."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


FISCHER_CONSTANT = 0.011
# Fischer et al. (1979) coefficient in K = 0.011 v^2 w^2 / (d u*).
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
        K (m^2/s), 0 where depth, ustar, or velocity is 0. (The Fischer model is not masked on
        flow_out: a dry reach already has depth or ustar 0. Only `dispersion_coefficient`'s
        "constant" model masks on flow_out, since its K does not come from the hydraulics.)
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
    background: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Per-reach dispersion coefficient for the configured model.

    Args:
        fields: per-reach hydraulics with keys velocity, depth, width, ustar, flow_out.
        model: "fischer", "constant", or "none".
        scale: Fischer multiplier.
        cap: Fischer upper bound (m^2/s).
        value: constant K (m^2/s) for the "constant" model.
        background: K (m^2/s) added on every reach with flow_out > 0, for every model including
            "none" (the analogue of the 2D/3D solver's ``lev``).

    Returns:
        K per reach (m^2/s); 0 where depth, ustar or velocity is 0 (and, for the constant
        model, where flow_out is 0), plus ``background`` where flow_out > 0.

    Raises:
        ValueError: unknown model, or "constant" without a value.
    """
    n = np.asarray(fields["velocity"]).shape[0]
    if model == "none":
        k = np.zeros(n, dtype=np.float64)
    elif model == "constant":
        if value is None:
            raise ValueError("dispersion model 'constant' requires a value")
        k = np.full(n, float(value), dtype=np.float64)
        k[np.asarray(fields["flow_out"]) <= 0.0] = 0.0
    elif model == "fischer":
        k = fischer_coefficient(
            fields["velocity"], fields["depth"], fields["width"], fields["ustar"], scale=scale, cap=cap
        )
    else:
        raise ValueError(f"unknown dispersion model {model!r}; expected one of {DISPERSION_MODELS}")
    if background != 0.0:
        k += np.where(np.asarray(fields["flow_out"]) > 0.0, float(background), 0.0)
    return k
