"""Configuration for the 1D network particle solver."""

from __future__ import annotations

import dataclasses
import datetime as dtm
import pathlib
import sys
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .dispersion import DISPERSION_MODELS
from .provider import INTERPOLATIONS


if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SOURCE_FORMS = ("slug", "loading", "concentration")
DTYPES = ("float32", "float64")


def parse_datetime(value: str | dtm.datetime | np.datetime64) -> np.datetime64:
    """Parse an ISO string, datetime, or datetime64 into datetime64[ns].

    Args:
        value: an ISO 8601 string, a `datetime.datetime`, or a `numpy.datetime64`.

    Returns:
        The value as `numpy.datetime64` with nanosecond resolution.
    """
    if isinstance(value, np.datetime64):
        return cast("np.datetime64", value.astype("datetime64[ns]"))
    if isinstance(value, dtm.datetime):
        return np.datetime64(value.replace(tzinfo=None), "ns")
    return np.datetime64(str(value), "ns")


@dataclasses.dataclass(frozen=True)
class DispersionConfig:
    """Longitudinal dispersion settings.

    Args:
        model: "fischer", "constant", or "none".
        scale: multiplier on the Fischer coefficient.
        cap: upper bound on K (m^2/s).
        value: K (m^2/s) for the constant model.
    """

    model: str = "fischer"
    scale: float = 1.0
    cap: float | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        """Validate the dispersion settings.

        Raises:
            ValueError: an unknown model, a non-positive scale or cap, or the
                "constant" model without a non-negative value.
        """
        if self.model not in DISPERSION_MODELS:
            raise ValueError(f"dispersion model must be one of {DISPERSION_MODELS}, got {self.model!r}")
        if self.scale <= 0.0:
            raise ValueError("dispersion scale must be positive")
        if self.cap is not None and self.cap <= 0.0:
            raise ValueError("dispersion cap must be positive")
        if self.model == "constant" and (self.value is None or self.value < 0.0):
            raise ValueError("dispersion model 'constant' requires a non-negative value")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DispersionConfig:
        """Build from a mapping; unknown keys raise.

        Args:
            d: mapping of dispersion config fields.

        Returns:
            The constructed `DispersionConfig`.

        Raises:
            ValueError: `d` contains a key that is not a `DispersionConfig` field.
        """
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown dispersion keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of the dispersion settings.

        Returns:
            A JSON-safe dict of the dispersion fields.
        """
        return dataclasses.asdict(self)


_DATETIME_KEYS = ("time", "start", "end")


def _jsonify_datetime(value: Any) -> Any:
    """Normalize a `datetime.datetime`/`datetime.date`/`np.datetime64` to an ISO string; pass through otherwise.

    tomllib parses an unquoted TOML datetime as `datetime.datetime`, which `json.dumps` (used by
    `NetworkConfig.to_dict()` consumers such as `run.py`'s output attrs) cannot serialize.
    """
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[s]"))
    if isinstance(value, dtm.date | dtm.datetime):
        return value.isoformat()
    return value


def _validate_source(i: int, row: Mapping[str, Any], particle_mass: float | None) -> dict[str, Any]:
    """Validate one source row.

    Args:
        i: index of the source in the sources sequence (for error messages).
        row: the raw source mapping.
        particle_mass: the config's global particle mass, if any.

    Returns:
        A plain dict copy of the validated row, with any `datetime.datetime`/`datetime.date`/
        `np.datetime64` values in ``time``, ``start``, ``end``, or ``curve`` pairs normalized to ISO
        strings so the row is JSON-safe by construction.

    Raises:
        ValueError: the row is missing a reach_id, has an invalid form, sets
            both s and s_frac, has a non-positive particles count, lacks both
            particles and a global particle_mass, or is missing a key its
            form requires.
    """
    if "reach_id" not in row:
        raise ValueError(f"sources[{i}] needs a reach_id")
    form = row.get("form")
    if form not in SOURCE_FORMS:
        raise ValueError(f"sources[{i}] form must be one of {SOURCE_FORMS}, got {form!r}")
    if "s" in row and "s_frac" in row:
        raise ValueError(f"sources[{i}] may set s or s_frac, not both")
    if "particles" in row and int(row["particles"]) < 1:
        raise ValueError(f"sources[{i}] particles must be a positive integer")
    if "particles" not in row and particle_mass is None:
        raise ValueError(f"sources[{i}] needs particles, or set a global particle_mass")
    if form == "slug" and not {"time", "mass"} <= set(row):
        raise ValueError(f"sources[{i}] form 'slug' needs time and mass")
    if form == "loading" and not ({"rate"} <= set(row) or {"curve"} <= set(row)):
        raise ValueError(f"sources[{i}] form 'loading' needs rate or curve")
    if form == "concentration" and not ({"value"} <= set(row) or {"curve"} <= set(row)):
        raise ValueError(f"sources[{i}] form 'concentration' needs value or curve")
    out = dict(row)
    for key in _DATETIME_KEYS:
        if key in out:
            out[key] = _jsonify_datetime(out[key])
    if "curve" in out:
        out["curve"] = [(_jsonify_datetime(t), v) for t, v in out["curve"]]
    return out


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    """Settings for one network particle run (see the [network] TOML table)."""

    hydraulics_file: str
    sources: tuple[dict[str, Any], ...]
    interpolation: str = "linear"
    reach_subset: tuple[int, ...] | dict[str, int] | None = None
    dtype: str = "float64"
    start_time: np.datetime64 | None = None
    end_time: np.datetime64 | None = None
    dt: float = 900.0
    output_interval: float = 3600.0
    dispersion: DispersionConfig = dataclasses.field(default_factory=DispersionConfig)
    particle_mass: float | None = None
    mass_units: str = "kg"
    max_hops: int = 1000
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the config's fields.

        Delegates to `_validate_scalars`, `_validate_source`,
        `_normalize_reach_subset`, and `_normalize_times`, any of which may
        raise `ValueError` on invalid input.
        """
        self._validate_scalars()
        object.__setattr__(
            self, "sources", tuple(_validate_source(i, r, self.particle_mass) for i, r in enumerate(self.sources))
        )
        self._normalize_reach_subset()
        self._normalize_times()

    def _validate_scalars(self) -> None:
        """Validate the simple scalar fields.

        Raises:
            ValueError: interpolation, dtype, dt, output_interval,
                particle_mass, max_hops, or sources fails validation.
        """
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {self.interpolation!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        ratio = self.output_interval / self.dt
        if self.output_interval <= 0.0 or abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("output_interval must be a positive integer multiple of dt")
        if self.particle_mass is not None and self.particle_mass <= 0.0:
            raise ValueError("particle_mass must be positive")
        if self.max_hops < 1:
            raise ValueError("max_hops must be at least 1")
        if not self.sources:
            raise ValueError("sources must contain at least one source")

    def _normalize_reach_subset(self) -> None:
        """Validate and normalize `reach_subset` in place.

        Raises:
            ValueError: `reach_subset` is a mapping with keys other than
                `{"outlet"}`.
        """
        if isinstance(self.reach_subset, Mapping):
            if set(self.reach_subset) != {"outlet"}:
                raise ValueError("reach_subset mapping must be {'outlet': reach_id}")
            # Frozen dataclass: direct attribute assignment is not possible here.
            object.__setattr__(self, "reach_subset", {"outlet": int(self.reach_subset["outlet"])})  # noqa: PLC2801
        elif self.reach_subset is not None:
            object.__setattr__(self, "reach_subset", tuple(int(r) for r in self.reach_subset))  # noqa: PLC2801

    def _normalize_times(self) -> None:
        """Parse `start_time`/`end_time` and check their ordering.

        Raises:
            ValueError: both are set and `start_time` is not before `end_time`.
        """
        for name in ("start_time", "end_time"):
            v = getattr(self, name)
            if v is not None:
                # Frozen dataclass: direct attribute assignment is not possible here.
                object.__setattr__(self, name, parse_datetime(v))  # noqa: PLC2801
        if self.start_time is not None and self.end_time is not None and self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> NetworkConfig:
        """Build from a plain mapping such as the [network] table; unknown keys raise.

        Args:
            d: mapping of network config fields.

        Returns:
            The constructed `NetworkConfig`.

        Raises:
            ValueError: `d` contains a key that is not a `NetworkConfig` field.
        """
        data = dict(d)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown network config keys: {sorted(unknown)}")
        disp = data.get("dispersion", {})
        data["dispersion"] = disp if isinstance(disp, DispersionConfig) else DispersionConfig.from_dict(disp)
        data["sources"] = tuple(dict(r) for r in data.get("sources", ()))
        return cls(**data)

    @classmethod
    def from_toml(cls, path: str | pathlib.Path) -> NetworkConfig:
        """Read the [network] table of a TOML file.

        Args:
            path: path to the TOML file.

        Returns:
            The constructed `NetworkConfig`.

        Raises:
            ValueError: the file has no [network] table.
        """
        with pathlib.Path(path).open("rb") as f:
            doc = tomllib.load(f)
        if "network" not in doc:
            raise ValueError(f"{path} has no [network] table")
        return cls.from_dict(doc["network"])

    @classmethod
    def coerce(cls, obj: NetworkConfig | Mapping[str, Any] | str | pathlib.Path) -> NetworkConfig:
        """Accept a NetworkConfig, a mapping, or a TOML path.

        Args:
            obj: an existing `NetworkConfig`, a plain mapping, or a path to a
                TOML file.

        Returns:
            `obj` itself if already a `NetworkConfig`, otherwise a config
            built from it.
        """
        if isinstance(obj, NetworkConfig):
            return obj
        if isinstance(obj, Mapping):
            return cls.from_dict(obj)
        return cls.from_toml(obj)

    def resolve_times(self, times: npt.NDArray[np.datetime64]) -> tuple[np.datetime64, np.datetime64]:
        """Start and end of the run, defaulting to the provider's first and last timestamps.

        Args:
            times: the hydraulics provider's time axis, ascending.

        Returns:
            The resolved (start, end) times.

        Raises:
            ValueError: the run window lies outside the provider's time axis,
                or start_time is not before end_time.
        """
        start = times[0] if self.start_time is None else self.start_time
        end = times[-1] if self.end_time is None else self.end_time
        if start < times[0] or start > times[-1]:
            raise ValueError(f"start_time {start} is outside the hydraulics time axis {times[0]}..{times[-1]}")
        if end < times[0] or end > times[-1]:
            raise ValueError(f"end_time {end} is outside the hydraulics time axis {times[0]}..{times[-1]}")
        if start >= end:
            raise ValueError("start_time must be before end_time")
        return start, end

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (datetimes as ISO strings) that from_dict accepts.

        Returns:
            A JSON-safe dict of the config's fields.
        """
        d = dataclasses.asdict(self)
        d["sources"] = [dict(r) for r in self.sources]
        d["dispersion"] = self.dispersion.to_dict()
        for name in ("start_time", "end_time"):
            if d[name] is not None:
                d[name] = str(np.datetime64(d[name], "s"))
        if isinstance(self.reach_subset, tuple):
            d["reach_subset"] = list(self.reach_subset)
        return d


NETWORK_TEMPLATE = """# fluvial-particle network solver settings
# All keys live in the [network] table. Times are ISO datetimes or seconds from start_time.

[network]
hydraulics_file = "drb_network_hydraulics.nc"   # pywatershed network hydraulics export
interpolation = "linear"      # "linear" between daily values, or "hold"
dtype = "float64"             # "float32" for very large networks
# reach_subset = [4205, 4204]  # reach ids to keep; or {outlet = 4205} for everything upstream of one reach
# start_time = "1979-03-01"    # default: first timestamp in the file
# end_time = "1979-04-01"      # default: last timestamp in the file
dt = 900.0                    # solver step (s)
output_interval = 3600.0      # output every n*dt seconds
particle_mass = 1.0           # mass per particle; sources may override with particles = N
mass_units = "kg"             # label only; concentrations are reported in mass_units m-3
max_hops = 1000
# seed = 42

[network.dispersion]
model = "fischer"             # "fischer", "constant", or "none"
scale = 1.0                   # multiplier on the Fischer coefficient
# cap = 1000.0                # optional upper bound (m2/s)
# value = 10.0                # K for model = "constant"

# Sources: a slug (instantaneous mass), a loading (mass rate), or a concentration at the release point.
[[network.sources]]
reach_id = 1234
form = "slug"
time = 0.0
mass = 1000.0

[[network.sources]]
reach_id = 2345
s_frac = 0.5                  # release mid-reach (or s = meters from the upstream end)
form = "loading"
rate = 0.01                   # mass_units per second
start = 0.0
end = 86400.0
# spacing = "even"            # or "poisson"

[[network.sources]]
reach_id = 3456
form = "concentration"        # mass_units per m3 at the release point, times the reach flow
curve = [[0.0, 0.0], [3600.0, 5.0], [7200.0, 0.0]]
particles = 500               # fixed count for this source
"""


def get_network_config_template() -> str:
    """Return the commented TOML template for a network run.

    Returns:
        The template text.
    """
    return NETWORK_TEMPLATE
