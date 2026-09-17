"""Particle models for the network solver: declared per-particle state and the model registry."""

from __future__ import annotations

import dataclasses
import importlib
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..random_walk import reflect_interval
from .solver import ACTIVE, SETTLED, FloatArray, Hydraulics, IntArray, NetworkSolver
from .vertical import SUBSTEP_FRACTION, VerticalProfiles, deposition_probability, substep_count, substep_counts


STATE_KINDS = ("extensive", "intensive")


@dataclasses.dataclass(frozen=True)
class StateVar:
    """One per-particle state array a model declares in its ``STATE`` class attribute.

    The base solver allocates the array on construction, the writer creates an output variable for
    it when ``output`` is True, and ``NetworkResults.positions`` returns it. Models never touch the
    writer or the results.

    Args:
        name: variable name in the solver's ``state`` mapping and in the output file.
        dtype: numpy dtype string.
        shape: per-particle shape, ``()`` for a scalar or ``(k,)`` for a vector.
        fill: value the array is filled with at allocation (NaN for floats by default).
        output: write the variable to the particle file.
        units: CF units string.
        long_name: CF long name.
        kind: ``"extensive"`` (a carried mass, summed per bin) or ``"intensive"`` (a property of the
            parcel, mass-weighted per bin), or None for state that is not a constituent.
        dim: name of the ``(k,)`` dimension in the output file; required for a vector state.
        labels: coordinate labels for the ``(k,)`` dimension, one per component.
    """

    name: str
    dtype: str = "f8"
    shape: tuple[int, ...] = ()
    fill: float | int = np.nan
    output: bool = True
    units: str = ""
    long_name: str = ""
    kind: str | None = None
    dim: str | None = None
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the declaration.

        Raises:
            ValueError: an unknown ``kind``, a shape with more than one axis, a vector state without
                ``dim``, or a ``labels`` length that does not match the vector length.
        """
        if self.kind is not None and self.kind not in STATE_KINDS:
            raise ValueError(f"StateVar {self.name!r}: kind must be one of {STATE_KINDS} or None, got {self.kind!r}")
        if len(self.shape) > 1:
            raise ValueError(f"StateVar {self.name!r}: shape must be () or (k,), got {self.shape}")
        if self.shape and self.dim is None:
            raise ValueError(f"StateVar {self.name!r}: a vector state needs a dim name")
        if self.shape and self.shape[0] < 1:
            raise ValueError(f"StateVar {self.name!r}: a vector state needs at least one component")
        if not self.shape and (self.dim is not None or self.labels is not None):
            raise ValueError(f"StateVar {self.name!r}: dim and labels apply to a vector state only")
        if self.labels is not None and (not self.shape or len(self.labels) != self.shape[0]):
            raise ValueError(f"StateVar {self.name!r}: labels must have one entry per component ({self.shape})")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as e:
            raise ValueError(f"StateVar {self.name!r}: {self.dtype!r} is not a numpy dtype") from e
        if dtype.kind in "iub":
            fill = float(self.fill)
            if not (np.isfinite(fill) and fill.is_integer()):
                raise ValueError(f"StateVar {self.name!r}: dtype {self.dtype!r} needs an explicit integer fill")
            if not np.can_cast(np.min_scalar_type(int(fill)), dtype):
                raise ValueError(f"StateVar {self.name!r}: fill {self.fill!r} does not fit dtype {self.dtype!r}")
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.labels is not None:
            object.__setattr__(self, "labels", tuple(self.labels))


# ---- drift model ---------------------------------------------------------------
DIEL_KEYS = ("amplitude", "period", "phase")


def _validate_initial_zeta(init: Any, zeta_min: float) -> str | float:
    """``"uniform"`` (over the whole column) or a number strictly inside ``(zeta_min, 1 - zeta_min)``.

    A fixed release inside the contact layer would be eligible to deposit on its first sub-step,
    and one at the surface clip has no meaning, so the number is kept out of both clipped bands.

    Raises:
        ValueError: any other string, or a number on or outside the domain.
    """
    if isinstance(init, str):
        if init != "uniform":
            raise ValueError(f"initial_zeta must be 'uniform' or a number, got {init!r}")
        return init
    value = float(init)
    if not zeta_min < value < 1.0 - zeta_min:
        raise ValueError(f"initial_zeta must lie inside (zeta_min, 1 - zeta_min), got {value}")
    return value


def _validate_diel(diel: Mapping[str, Any]) -> dict[str, float]:
    """The ``diel`` table: ``amplitude`` (m/s) and ``period`` (s) required, ``phase`` (rad) optional.

    Raises:
        ValueError: a missing or unknown key, a negative amplitude, or a non-positive period.
    """
    d = dict(diel)
    if set(d) - set(DIEL_KEYS) or not {"amplitude", "period"} <= set(d):
        raise ValueError(f"diel must be a table with keys amplitude and period (phase optional), got {d}")
    out = {"amplitude": float(d["amplitude"]), "period": float(d["period"]), "phase": float(d.get("phase", 0.0))}
    if out["amplitude"] < 0.0:
        raise ValueError("diel amplitude must be non-negative")
    if out["period"] <= 0.0:
        raise ValueError("diel period must be positive")
    return out


class DriftParticles(NetworkSolver):
    """Quasi-2D drift: a resolved vertical position per particle with settling, swimming and deposition.

    Each particle carries ``zeta = z / h`` (0 bed, 1 surface) on ``[0, 1]``, preserved across reach
    hops. Per step the vertical walk runs ``n_sub`` sub-steps, each an operator split of

    1. mixing with the configured profile (``[network.dispersion.vertical]``) by a kernel that leaves
       a uniform column exactly uniform at any step (``VerticalProfiles.mix``: a random rotation on
       the sphere for the parabolic profile, a reflected Gaussian step for a constant one);
    2. the vertical velocity ``w`` (positive down: ``settling_velocity - swim_velocity + diel``) as a
       displacement ``-w dt / h``, mirror-reflected at the surface and the bed;
    3. deposition: a particle whose shifted position, before reflection, is in the bed contact layer
       ``zeta < zeta_min`` (or below the bed) has made contact and deposits with the probability that
       reproduces a Robin bed condition with deposition velocity ``k_d``
       (``vertical.deposition_probability``), else reflects and keeps walking.

    The velocity factor handed to advection is the mean of ``u(zeta) / v`` over the sub-steps, with
    the log law evaluated at ``clip(zeta, zeta_min, 1 - zeta_min)`` and normalized to unit mean over
    the column, exactly as the shear-dispersion quadrature uses it.

    Parameters (``[network.particles]``, ``model = "drift"``): ``settling_velocity`` (m/s, down),
    ``swim_velocity`` (m/s, up), ``diel`` (``{amplitude, period, phase}``: ``w`` gains
    ``amplitude sin(2 pi t / period + phase)``), ``deposition_velocity`` (m/s; 0 is a reflecting
    bed), ``critical_ustar`` (m/s; Krone's factor ``max(0, 1 - (ustar / critical_ustar)^2)`` on the
    deposition velocity), ``zeta_min`` (velocity clip and contact-layer thickness), ``substep_fraction``
    (fraction of the column mixing time per sub-step), ``max_substeps``, ``initial_zeta``
    (``"uniform"`` or a number).
    """

    resolves_vertical = True
    STATE = (
        StateVar("zeta", units="1", long_name="relative elevation in the water column, 0 bed, 1 surface"),
        StateVar(
            "velocity_factor",
            units="1",
            long_name="mean of u(zeta)/v over the step's sub-steps; 1 for a particle that deposited this step",
            output=False,
        ),
    )
    PARAM_DEFAULTS: Mapping[str, Any] = {
        "settling_velocity": 0.0,
        "swim_velocity": 0.0,
        "diel": None,
        "deposition_velocity": 0.0,
        "critical_ustar": None,
        "zeta_min": 0.001,
        "substep_fraction": SUBSTEP_FRACTION,
        "max_substeps": 1000,
        "initial_zeta": "uniform",
    }

    @classmethod
    def validate_params(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the drift parameters and fill in defaults.

        Args:
            params: the ``[network.particles]`` table without ``model``.

        Returns:
            The complete parameter dict.

        Raises:
            ValueError: an unknown key, a value out of range, or a malformed ``diel`` table.
        """
        unknown = set(params) - set(cls.PARAM_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown drift particle parameters: {sorted(unknown)}")
        p: dict[str, Any] = {**cls.PARAM_DEFAULTS, **params}
        for key in ("settling_velocity", "swim_velocity", "deposition_velocity"):
            p[key] = float(p[key])
            if p[key] < 0.0:
                raise ValueError(f"{key} must be non-negative (use the opposite key for the other direction)")
        if p["critical_ustar"] is not None:
            p["critical_ustar"] = float(p["critical_ustar"])
            if p["critical_ustar"] <= 0.0:
                raise ValueError("critical_ustar must be positive")
        p["zeta_min"] = float(p["zeta_min"])
        if not 0.0 < p["zeta_min"] < 0.5:
            raise ValueError("zeta_min must lie in (0, 0.5)")
        p["max_substeps"] = int(p["max_substeps"])
        if p["max_substeps"] < 1:
            raise ValueError("max_substeps must be at least 1")
        p["substep_fraction"] = float(p["substep_fraction"])
        if p["substep_fraction"] <= 0.0:
            raise ValueError("substep_fraction must be positive")
        p["initial_zeta"] = _validate_initial_zeta(p["initial_zeta"], p["zeta_min"])
        if p["diel"] is not None:
            p["diel"] = _validate_diel(p["diel"])
        return p

    def __init__(self, *args: Any, params: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """Construct the base solver, then the vertical profiles and the shear table."""
        super().__init__(*args, params=params, **kwargs)
        self.profiles = VerticalProfiles(self.dispersion.vertical, self.params["zeta_min"])
        self._shear_table = self.profiles if self._shear_correction_active() else None
        self.last_substeps = 0
        self.steps_at_cap = 0  # steps whose sub-step count hit max_substeps
        self.clipped_contacts = 0  # bed contacts whose deposition probability was clipped at 1
        self._warned_cap = False
        self._warned_clip = False

    # ---- parameters as functions ----------------------------------------------
    def _vertical_velocity(self, t_mid: float) -> float:
        """``w`` (m/s, positive down) at solver time ``t_mid``: settling - swimming + the diel term."""
        w = self.params["settling_velocity"] - self.params["swim_velocity"]
        diel = self.params["diel"]
        if diel is not None:
            w += diel["amplitude"] * np.sin(2.0 * np.pi * t_mid / diel["period"] + diel["phase"])
        return float(w)

    def _deposition_velocity(self, ustar: FloatArray) -> FloatArray:
        """``k_d`` per particle: the deposition velocity times Krone's factor when ``critical_ustar`` is set."""
        k_d = np.full(ustar.shape, self.params["deposition_velocity"])
        crit = self.params["critical_ustar"]
        if crit is not None:
            k_d *= np.maximum(0.0, 1.0 - (ustar / crit) ** 2)
        return k_d

    # ---- hooks -------------------------------------------------------------------
    def on_release(self, idx: IntArray, h: Hydraulics) -> None:  # noqa: ARG002
        """Draw the initial relative elevation of the particles released this step."""
        init = self.params["initial_zeta"]
        if init == "uniform":
            self._state["zeta"][idx] = self.rng.uniform(0.0, 1.0, idx.size)
        else:
            self._state["zeta"][idx] = float(init)

    def behave(self, h: Hydraulics, tau: FloatArray, t: float, dt: float) -> FloatArray | None:
        """Run the sub-stepped vertical walk for the active particles; settle those that deposit.

        Returns:
            The per-particle velocity factor (mean of ``u(zeta) / v`` over the sub-steps), or None
            when no particle is active.
        """
        idx = np.nonzero(self._status == ACTIVE)[0]
        if idx.size == 0:
            return None
        r = self._reach[idx].astype(np.int64)
        ustar = np.asarray(h["ustar"], dtype=np.float64)[r]
        depth = np.asarray(h["depth"], dtype=np.float64)[r]
        v = np.asarray(h["velocity"], dtype=np.float64)[r]
        n_sub = substep_count(
            dt,
            self.profiles.kz_max(ustar, depth),
            depth,
            c=self.params["substep_fraction"],
            max_substeps=self.params["max_substeps"],
        )
        self.last_substeps = n_sub
        if n_sub >= self.params["max_substeps"]:
            self.steps_at_cap += 1
            if not self._warned_cap:
                self._warned_cap = True
                warnings.warn(
                    f"the vertical walk hit max_substeps = {n_sub} at t = {t:.0f} s (a shallow, high-shear reach "
                    "holds active particles); the within-step shear dispersion is under-resolved there",
                    UserWarning,
                    stacklevel=2,
                )
        w = self._vertical_velocity(t + 0.5 * dt)
        k_d = self._deposition_velocity(ustar)
        # A dry reach (depth 0) has nothing to walk in: its particles keep their zeta.
        wet = depth > 0.0
        inv_depth = np.where(wet, 1.0 / np.where(wet, depth, 1.0), 0.0)
        zeta = self._state["zeta"][idx].copy()
        fsum = np.zeros(idx.size)
        alive = np.ones(idx.size, dtype=bool)
        dts = tau[idx] / n_sub  # particles released mid-step walk only for their time budget
        zmin = self.params["zeta_min"]
        p_dep = deposition_probability(k_d, dts, zmin * depth, settling=w)
        check_bed = np.any(p_dep > 0.0)
        for _ in range(n_sub):
            a = np.nonzero(alive)[0]
            if a.size == 0:
                break
            za = self.profiles.mix(zeta[a], ustar[a], depth[a], dts[a], self.rng)
            if w != 0.0:
                za -= w * dts[a] * inv_depth[a]
            if check_bed:
                # contact: the shifted position, before reflection, is in the layer [0, zeta_min) or below
                contact = np.nonzero(za < zmin)[0]
                if contact.size:
                    b = a[contact]
                    n_clip = int((p_dep[b] >= 1.0).sum())
                    if n_clip:
                        self.clipped_contacts += n_clip
                        if not self._warned_clip:
                            self._warned_clip = True
                            warnings.warn(
                                f"the per-contact deposition probability clipped at 1 at t = {t:.0f} s: the bed is "
                                "absorbing there and the deposition rate depends on the sub-step count "
                                "(raise zeta_min or lower substep_fraction)",
                                UserWarning,
                                stacklevel=2,
                            )
                    deposited = b[self.rng.uniform(size=b.size) < p_dep[b]]
                    alive[deposited] = False
            if w != 0.0:
                za = reflect_interval(za, 0.0, 1.0)
            zeta[a] = za
            live = a[alive[a]]
            fsum[live] += self.profiles.velocity_factor(zeta[live], ustar[live], v[live])
        zeta[~alive] = 0.0  # on the bed
        self._state["zeta"][idx] = zeta
        factor = np.ones(self.n)
        factor[idx[alive]] = fsum[alive] / n_sub
        if not alive.all():
            # Deposited particles keep the s they had at the start of the step (a first-order timing
            # approximation, documented) and are excluded from this step's advection.
            self.terminate(idx[~alive], SETTLED, t + dt)
        self._state["velocity_factor"][idx] = factor[idx]
        return factor

    def diagnostics(self, h: Hydraulics) -> list[str]:
        """Sub-step count and shear correction for one hydraulics slice (the startup report)."""
        ustar = np.asarray(h["ustar"], dtype=np.float64)
        depth = np.asarray(h["depth"], dtype=np.float64)
        v = np.asarray(h["velocity"], dtype=np.float64)
        wet = np.asarray(h["flow_out"], dtype=np.float64) > 0.0
        cap = self.params["max_substeps"]
        counts = substep_counts(self.dt, self.profiles.kz_max(ustar, depth), depth, c=self.params["substep_fraction"])[
            wet
        ]
        n_cap = int((counts > cap).sum())
        lines = [
            f"  vertical walk: sub-steps per step (first slice, over wet reaches) median "
            f"{int(np.median(counts)) if counts.size else 0}, max {int(counts.max()) if counts.size else 0} "
            f"(cap {cap}); {n_cap} wet reaches at the cap; the run uses the max over reaches holding particles"
        ]
        k_d = self.params["deposition_velocity"]
        if k_d > 0.0 and counts.size:
            n_sub = min(int(counts.max()), cap)
            p = deposition_probability(
                self._deposition_velocity(ustar[wet]),
                self.dt / n_sub,
                self.params["zeta_min"] * depth[wet],
                settling=self._vertical_velocity(0.5 * self.dt),
            )
            note = "; clipped at 1 on some reaches: raise zeta_min or lower substep_fraction" if p.max() >= 1.0 else ""
            lines.append(
                f"  deposition: per-contact probability (at that sub-step) median {np.median(p):.3g}, "
                f"max {p.max():.3g}{note}"
            )
        if self._shear_table is None:
            lines.append("  shear correction: not active")
        else:
            c = self._shear_table.shear_coefficient(ustar, v, depth)[wet]
            lines.append(
                f"  shear correction: active, coefficient c in [{c.min():.2f}, {c.max():.2f}] "
                f"(K_shear = c ustar h removed from the longitudinal K)"
                if c.size
                else "  shear correction: active"
            )
        return lines


# ---- registry ------------------------------------------------------------------
PARTICLE_MODELS: dict[str, type[NetworkSolver]] = {"passive": NetworkSolver, "drift": DriftParticles}


def resolve_model(name: str) -> type[NetworkSolver]:
    """The particle model class for a registry name or a ``"package.module:ClassName"`` path.

    Args:
        name: a key of ``PARTICLE_MODELS`` or a dotted import path with a colon before the class.

    Returns:
        The model class, a subclass of ``NetworkSolver``.

    Raises:
        KeyError: an unknown registry name (the message lists the known ones).
        ImportError: the module or the attribute of a dotted path cannot be imported.
        TypeError: the named object is not a subclass of ``NetworkSolver``.
    """
    if ":" in name:
        module_name, _, attr = name.partition(":")
        module = importlib.import_module(module_name)
        try:
            obj = getattr(module, attr)
        except AttributeError as e:
            raise ImportError(f"module {module_name!r} has no attribute {attr!r}") from e
    elif name in PARTICLE_MODELS:
        obj = PARTICLE_MODELS[name]
    else:
        raise KeyError(f"unknown particle model {name!r}; known models: {sorted(PARTICLE_MODELS)}")
    if not (isinstance(obj, type) and issubclass(obj, NetworkSolver)):
        raise TypeError(f"particle model {name!r} must be a subclass of NetworkSolver, got {obj!r}")
    return obj
