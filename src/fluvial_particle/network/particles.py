"""Particle models for the network solver: declared per-particle state and the model registry."""

from __future__ import annotations

import dataclasses

import numpy as np


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
        if self.labels is not None and (not self.shape or len(self.labels) != self.shape[0]):
            raise ValueError(f"StateVar {self.name!r}: labels must have one entry per component ({self.shape})")
