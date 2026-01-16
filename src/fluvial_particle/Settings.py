"""Settings file, subclass of dictionary."""

from __future__ import annotations

import pathlib
import sys
from typing import Any


# Use tomllib (Python 3.11+) or tomli (backport)
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .FallingParticles import FallingParticles
from .LarvalParticles import LarvalBotParticles, LarvalParticles, LarvalTopParticles
from .Particles import Particles


# Registry mapping particle type names to classes
# New Particles subclasses must be added here before they can be used
PARTICLE_REGISTRY: dict[str, type[Particles]] = {
    "Particles": Particles,
    "FallingParticles": FallingParticles,
    "LarvalParticles": LarvalParticles,
    "LarvalTopParticles": LarvalTopParticles,
    "LarvalBotParticles": LarvalBotParticles,
}

# Copy for use in exec() - exec() modifies its globals dict, so we can't use PARTICLE_REGISTRY directly
global_dict = dict(PARTICLE_REGISTRY)


class Settings(dict):
    """Handler class to user defined parameters. Allows us to check a users input parameters in the backend."""

    def __init__(self, **kwargs):
        """Check that options file has all required keys."""
        missing = [x for x in self.required_keys if x not in kwargs]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} from the user parameter file")

        for key, value in kwargs.items():
            self[key] = value

    @property
    def required_keys(self):
        """Attributes required in the options file.

        Returns:
            list(str): list of the required keys: SimTime, dt, Track3D, PrintAtTick, file_name_2d, file_name_3d,
            NumPart, StartLoc, field_map_2d, field_map_3d
        """
        return (
            "SimTime",
            "dt",
            "Track3D",
            "PrintAtTick",
            "file_name_2d",
            "file_name_3d",
            "NumPart",
            "StartLoc",
            "field_map_2d",
            "field_map_3d",
        )

    @classmethod
    def read(cls, filename: str | pathlib.Path) -> Settings:
        """Load user parameters from options file.

        Supports both Python (.py) and TOML (.toml) configuration files.
        Format is detected by file extension.

        Args:
            filename: Path to the options file (.py or .toml)

        Returns:
            Settings: dict-like object with user parameters stored as key: value pairs

        Raises:
            ValueError: If file extension is not .py or .toml
        """
        path = pathlib.Path(filename)
        suffix = path.suffix.lower()

        if suffix == ".toml":
            return cls._read_toml(path)
        if suffix == ".py":
            return cls._read_python(path)
        raise ValueError(f"Unsupported settings file format: {suffix}. Use .toml (recommended) or .py")

    @classmethod
    def _read_python(cls, path: pathlib.Path) -> Settings:
        """Load settings from a Python file (legacy format)."""
        options: dict[str, Any] = {}
        with path.open(encoding="utf-8") as f:
            content = "\n".join(f.readlines())
            exec(content, global_dict, options)  # noqa: S102 # nosec B102 - Intentional exec for user config

        return cls(**options)

    @classmethod
    def _read_toml(cls, path: pathlib.Path) -> Settings:
        """Load settings from a TOML file.

        TOML files use a nested structure that gets flattened to match
        the internal settings format expected by the simulation.
        """
        with path.open("rb") as f:
            toml_config = tomllib.load(f)

        # Convert nested TOML structure to flat settings dict
        options = cls._flatten_toml_config(toml_config)

        return cls(**options)

    @classmethod
    def _flatten_toml_config(cls, config: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0912
        """Convert nested TOML config to flat settings dict.

        TOML structure:
            [simulation]
            time = 60.0           -> SimTime
            dt = 0.25             -> dt
            print_interval = 20.0 -> PrintAtTick

            [particles]
            type = "Particles"    -> ParticleType (via registry lookup)
            count = 100           -> NumPart
            start_location = [x,y,z] -> StartLoc

            [grid]
            track_3d = true       -> Track3D
            file_2d = "..."       -> file_name_2d
            file_3d = "..."       -> file_name_3d

            [grid.field_map_2d]
            bed_elevation = "..." -> field_map_2d dict

        Returns:
            Flat dict with internal key names.
        """
        result: dict[str, Any] = {}

        # [simulation] section
        if "simulation" in config:
            sim = config["simulation"]
            if "time" in sim:
                result["SimTime"] = sim["time"]
            if "dt" in sim:
                result["dt"] = sim["dt"]
            if "print_interval" in sim:
                result["PrintAtTick"] = sim["print_interval"]

        # [particles] section
        if "particles" in config:
            particles = config["particles"]
            if "type" in particles:
                type_name = particles["type"]
                if type_name not in PARTICLE_REGISTRY:
                    available = ", ".join(PARTICLE_REGISTRY.keys())
                    raise ValueError(f"Unknown particle type: '{type_name}'. Available types: {available}")
                result["ParticleType"] = PARTICLE_REGISTRY[type_name]
            if "count" in particles:
                result["NumPart"] = particles["count"]
            if "start_location" in particles:
                loc = particles["start_location"]
                result["StartLoc"] = tuple(loc) if isinstance(loc, list) else loc
            if "start_depth_fraction" in particles:
                result["startfrac"] = particles["start_depth_fraction"]
            if "start_time" in particles:
                result["PartStartTime"] = particles["start_time"]

            # [particles.physics] subsection
            if "physics" in particles:
                physics = particles["physics"]
                if "beta" in physics:
                    beta = physics["beta"]
                    result["beta"] = tuple(beta) if isinstance(beta, list) else beta
                if "lev" in physics:
                    result["lev"] = physics["lev"]
                if "min_depth" in physics:
                    result["min_depth"] = physics["min_depth"]
                if "vertical_bound" in physics:
                    result["vertbound"] = physics["vertical_bound"]

            # [particles.falling] subsection (FallingParticles)
            if "falling" in particles:
                falling = particles["falling"]
                if "radius" in falling:
                    result["radius"] = falling["radius"]
                if "density" in falling:
                    result["rho"] = falling["density"]
                if "c1" in falling:
                    result["c1"] = falling["c1"]
                if "c2" in falling:
                    result["c2"] = falling["c2"]

            # [particles.larval] subsection (LarvalParticles)
            if "larval" in particles:
                larval = particles["larval"]
                if "amplitude" in larval:
                    result["amp"] = larval["amplitude"]
                if "period" in larval:
                    result["period"] = larval["period"]

        # [grid] section
        if "grid" in config:
            grid = config["grid"]
            if "track_3d" in grid:
                result["Track3D"] = 1 if grid["track_3d"] else 0
            if "file_2d" in grid:
                result["file_name_2d"] = grid["file_2d"]
            if "file_3d" in grid:
                result["file_name_3d"] = grid["file_3d"]

            # [grid.field_map_2d] subsection
            if "field_map_2d" in grid:
                result["field_map_2d"] = dict(grid["field_map_2d"])

            # [grid.field_map_3d] subsection
            if "field_map_3d" in grid:
                result["field_map_3d"] = dict(grid["field_map_3d"])

            # [grid.friction] subsection
            if "friction" in grid:
                friction = grid["friction"]
                if "manning_n" in friction:
                    result["manning_n"] = friction["manning_n"]
                if "chezy_c" in friction:
                    result["chezy_c"] = friction["chezy_c"]
                if "darcy_f" in friction:
                    result["darcy_f"] = friction["darcy_f"]
                if "water_density" in friction:
                    result["water_density"] = friction["water_density"]
                if "ustar_method" in friction:
                    result["ustar_method"] = friction["ustar_method"]

            # [grid.time_varying] subsection
            if "time_varying" in grid:
                tv = grid["time_varying"]
                if tv.get("enabled", False):
                    result["time_dependent"] = True
                    if "file_pattern_2d" in tv:
                        result["file_pattern_2d"] = tv["file_pattern_2d"]
                    if "file_pattern_3d" in tv:
                        result["file_pattern_3d"] = tv["file_pattern_3d"]
                    if "start_index" in tv:
                        result["grid_start_index"] = tv["start_index"]
                    if "end_index" in tv:
                        result["grid_end_index"] = tv["end_index"]
                    if "dt" in tv:
                        result["grid_dt"] = tv["dt"]
                    if "start_time" in tv:
                        result["grid_start_time"] = tv["start_time"]
                    if "interpolation" in tv:
                        result["grid_interpolation"] = tv["interpolation"]

        # [output] section
        if "output" in config:
            output = config["output"]
            if "vtp" in output:
                result["output_vtp"] = output["vtp"]
            if "hdf5_kwargs" in output:
                result["hdf5_dataset_kwargs"] = output["hdf5_kwargs"]

        return result
