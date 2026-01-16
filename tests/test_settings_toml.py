"""Tests for TOML settings file support."""

import pathlib
import tempfile

import pytest

from fluvial_particle.Particles import Particles
from fluvial_particle.Settings import PARTICLE_REGISTRY, Settings


class TestParticleRegistry:
    """Tests for the particle type registry."""

    def test_registry_contains_all_particle_types(self):
        """All particle types should be in the registry."""
        expected = {
            "Particles",
            "FallingParticles",
            "LarvalParticles",
            "LarvalTopParticles",
            "LarvalBotParticles",
        }
        assert set(PARTICLE_REGISTRY.keys()) == expected

    def test_registry_values_are_classes(self):
        """Registry values should be class objects."""
        for name, cls in PARTICLE_REGISTRY.items():
            assert isinstance(cls, type), f"{name} should be a class"


class TestTOMLSettings:
    """Tests for TOML settings file loading."""

    @pytest.fixture
    def toml_settings_path(self):
        """Path to the test TOML settings file."""
        return pathlib.Path("tests/data/user_options_test.toml")

    @pytest.fixture
    def py_settings_path(self):
        """Path to a test Python settings file."""
        return pathlib.Path("tests/data/user_options_straight_test_vts.py")

    def test_read_toml_file(self, toml_settings_path):
        """Should successfully read a TOML settings file."""
        settings = Settings.read(toml_settings_path)
        assert isinstance(settings, Settings)

    def test_toml_simulation_settings(self, toml_settings_path):
        """TOML simulation settings should be correctly parsed."""
        settings = Settings.read(toml_settings_path)
        assert settings["SimTime"] == 60.0
        assert settings["dt"] == 0.25
        assert settings["PrintAtTick"] == 20.0

    def test_toml_particle_settings(self, toml_settings_path):
        """TOML particle settings should be correctly parsed."""
        settings = Settings.read(toml_settings_path)
        assert settings["ParticleType"] == Particles
        assert settings["NumPart"] == 20
        assert settings["StartLoc"] == (5.0, 0.0, 9.5)
        assert settings["startfrac"] == 0.5

    def test_toml_particle_physics(self, toml_settings_path):
        """TOML particle physics settings should be correctly parsed."""
        settings = Settings.read(toml_settings_path)
        assert settings["beta"] == (0.067, 0.067, 0.067)
        assert settings["lev"] == 0.00025
        assert settings["min_depth"] == 0.02
        assert settings["vertbound"] == 0.01

    def test_toml_grid_settings(self, toml_settings_path):
        """TOML grid settings should be correctly parsed."""
        settings = Settings.read(toml_settings_path)
        assert settings["Track3D"] == 1
        assert settings["file_name_2d"] == "./tests/data/Result_straight_2d_1.vts"
        assert settings["file_name_3d"] == "./tests/data/Result_straight_3d_1_new.vts"

    def test_toml_field_maps(self, toml_settings_path):
        """TOML field maps should be correctly parsed as dicts."""
        settings = Settings.read(toml_settings_path)
        assert isinstance(settings["field_map_2d"], dict)
        assert settings["field_map_2d"]["bed_elevation"] == "Elevation"
        assert settings["field_map_2d"]["velocity"] == "Velocity"
        assert isinstance(settings["field_map_3d"], dict)
        assert settings["field_map_3d"]["velocity"] == "Velocity"

    def test_read_python_file_still_works(self, py_settings_path):
        """Python settings files should still work (backward compatibility)."""
        settings = Settings.read(py_settings_path)
        assert isinstance(settings, Settings)
        assert settings["SimTime"] == 60.0

    def test_unsupported_format_raises_error(self):
        """Unsupported file formats should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            path = pathlib.Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported settings file format"):
                Settings.read(path)
        finally:
            path.unlink()


class TestTOMLParticleTypes:
    """Tests for different particle types in TOML."""

    def test_falling_particles_type(self):
        """FallingParticles type should be correctly resolved."""
        toml_content = """
[simulation]
time = 60.0
dt = 0.25
print_interval = 20.0

[particles]
type = "FallingParticles"
count = 10
start_location = [5.0, 0.0, 9.5]

[particles.falling]
radius = 0.0005
density = 2650.0

[grid]
track_3d = true
file_2d = "./tests/data/Result_straight_2d_1.vts"
file_3d = "./tests/data/Result_straight_3d_1_new.vts"

[grid.field_map_2d]
bed_elevation = "Elevation"
shear_stress = "ShearStress (magnitude)"
velocity = "Velocity"
water_surface_elevation = "WaterSurfaceElevation"

[grid.field_map_3d]
velocity = "Velocity"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w", encoding="utf-8") as f:
            f.write(toml_content)
            path = pathlib.Path(f.name)

        try:
            settings = Settings.read(path)
            from fluvial_particle.FallingParticles import FallingParticles

            assert settings["ParticleType"] == FallingParticles
            assert settings["radius"] == 0.0005
            assert settings["rho"] == 2650.0
        finally:
            path.unlink()

    def test_invalid_particle_type_raises_error(self):
        """Invalid particle type should raise ValueError."""
        toml_content = """
[simulation]
time = 60.0
dt = 0.25
print_interval = 20.0

[particles]
type = "NonExistentParticle"
count = 10
start_location = [5.0, 0.0, 9.5]

[grid]
track_3d = true
file_2d = "./tests/data/Result_straight_2d_1.vts"
file_3d = "./tests/data/Result_straight_3d_1_new.vts"

[grid.field_map_2d]
bed_elevation = "Elevation"
shear_stress = "ShearStress (magnitude)"
velocity = "Velocity"
water_surface_elevation = "WaterSurfaceElevation"

[grid.field_map_3d]
velocity = "Velocity"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w", encoding="utf-8") as f:
            f.write(toml_content)
            path = pathlib.Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unknown particle type"):
                Settings.read(path)
        finally:
            path.unlink()


class TestGetDefaultConfig:
    """Tests for get_default_config helper function."""

    def test_get_default_config_returns_dict(self):
        """get_default_config should return a nested dict."""
        from fluvial_particle import get_default_config

        config = get_default_config()
        assert isinstance(config, dict)
        assert "simulation" in config
        assert "particles" in config
        assert "grid" in config

    def test_get_default_config_is_deep_copy(self):
        """Modifying returned config should not affect the default."""
        from fluvial_particle import get_default_config

        config1 = get_default_config()
        config2 = get_default_config()

        config1["particles"]["count"] = 999

        assert config2["particles"]["count"] == 100  # Original value


class TestSaveConfig:
    """Tests for save_config helper function."""

    def test_save_config_creates_file(self):
        """save_config should create a TOML file."""
        from fluvial_particle import get_default_config, save_config

        config = get_default_config()

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            path = pathlib.Path(f.name)

        # Remove the file so save_config can create it
        path.unlink()

        try:
            save_config(config, path)
            assert path.exists()

            # Verify it's valid TOML
            import sys

            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib

            with pathlib.Path(path).open("rb") as f:
                loaded = tomllib.load(f)
            assert "simulation" in loaded
            assert "particles" in loaded
        finally:
            if path.exists():
                path.unlink()


class TestTOMLTimeVarying:
    """Tests for time-varying grid settings in TOML."""

    def test_time_varying_disabled_by_default(self):
        """Time-varying should not be set if not enabled."""
        toml_content = """
[simulation]
time = 60.0
dt = 0.25
print_interval = 20.0

[particles]
type = "Particles"
count = 10
start_location = [5.0, 0.0, 9.5]

[grid]
track_3d = true
file_2d = "./tests/data/Result_straight_2d_1.vts"
file_3d = "./tests/data/Result_straight_3d_1_new.vts"

[grid.field_map_2d]
bed_elevation = "Elevation"
shear_stress = "ShearStress (magnitude)"
velocity = "Velocity"
water_surface_elevation = "WaterSurfaceElevation"

[grid.field_map_3d]
velocity = "Velocity"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w", encoding="utf-8") as f:
            f.write(toml_content)
            path = pathlib.Path(f.name)

        try:
            settings = Settings.read(path)
            assert "time_dependent" not in settings
        finally:
            path.unlink()


class TestRunSimulationWithDict:
    """Tests for run_simulation with dict config."""

    def test_run_simulation_with_dict_config(self, tmp_path):
        """run_simulation should work with dict config directly."""
        from fluvial_particle import get_default_config, run_simulation

        config = get_default_config()
        config["particles"]["count"] = 5
        config["particles"]["start_location"] = [5.0, 0.0, 9.5]
        config["particles"]["start_depth_fraction"] = 0.5
        config["simulation"]["time"] = 1.0
        config["simulation"]["print_interval"] = 1.0
        config["grid"]["file_2d"] = "./tests/data/Result_straight_2d_1.vts"
        config["grid"]["file_3d"] = "./tests/data/Result_straight_3d_1_new.vts"

        output_dir = tmp_path / "output"
        results = run_simulation(config, output_dir, seed=42, quiet=True)

        assert results.num_particles == 5


class TestSaveConfigRoundTrip:
    """Tests for save_config round-trip compatibility."""

    def test_save_config_round_trip(self):
        """Saved config should be readable by Settings.read()."""
        from fluvial_particle import get_default_config, save_config

        config = get_default_config()
        config["simulation"]["time"] = 120.0
        config["particles"]["count"] = 50
        config["grid"]["file_2d"] = "./tests/data/Result_straight_2d_1.vts"
        config["grid"]["file_3d"] = "./tests/data/Result_straight_3d_1_new.vts"

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            path = pathlib.Path(f.name)

        path.unlink()  # Remove so save_config can create it

        try:
            save_config(config, path)
            settings = Settings.read(path)

            assert settings["SimTime"] == 120.0
            assert settings["NumPart"] == 50
            assert settings["file_name_2d"] == "./tests/data/Result_straight_2d_1.vts"
        finally:
            if path.exists():
                path.unlink()


class TestLarvalParticlesType:
    """Tests for LarvalParticles type in TOML."""

    def test_larval_particles_type(self):
        """LarvalParticles type should be correctly resolved."""
        toml_content = """
[simulation]
time = 60.0
dt = 0.25
print_interval = 20.0

[particles]
type = "LarvalParticles"
count = 10
start_location = [5.0, 0.0, 9.5]

[particles.larval]
amplitude = 0.15
period = 30.0

[grid]
track_3d = true
file_2d = "./tests/data/Result_straight_2d_1.vts"
file_3d = "./tests/data/Result_straight_3d_1_new.vts"

[grid.field_map_2d]
bed_elevation = "Elevation"
shear_stress = "ShearStress (magnitude)"
velocity = "Velocity"
water_surface_elevation = "WaterSurfaceElevation"

[grid.field_map_3d]
velocity = "Velocity"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w", encoding="utf-8") as f:
            f.write(toml_content)
            path = pathlib.Path(f.name)

        try:
            settings = Settings.read(path)
            from fluvial_particle.LarvalParticles import LarvalParticles

            assert settings["ParticleType"] == LarvalParticles
            assert settings["amp"] == 0.15
            assert settings["period"] == 30.0
        finally:
            path.unlink()


class TestSettingsFromDict:
    """Tests for Settings.from_dict() public API."""

    def test_from_dict_creates_settings(self):
        """Settings.from_dict() should create valid Settings object."""
        from fluvial_particle import get_default_config

        config = get_default_config()
        config["grid"]["file_2d"] = "./tests/data/Result_straight_2d_1.vts"
        config["grid"]["file_3d"] = "./tests/data/Result_straight_3d_1_new.vts"

        settings = Settings.from_dict(config)

        assert isinstance(settings, Settings)
        assert settings["SimTime"] == 60.0
        assert settings["NumPart"] == 100


class TestSettingsErrorHandling:
    """Tests for Settings error handling."""

    def test_no_extension_error(self):
        """File without extension should raise clear error."""
        with pytest.raises(ValueError, match="No file extension found"):
            Settings.read("settings_file_without_extension")

    def test_string_escaping_in_save_config(self):
        """Strings with special characters should be properly escaped."""
        from fluvial_particle import get_default_config, save_config

        config = get_default_config()
        config["grid"]["file_2d"] = './path/with "quotes" and \\ backslash.vts'

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            path = pathlib.Path(f.name)

        path.unlink()

        try:
            save_config(config, path)

            # Verify it's valid TOML that can be loaded
            import sys

            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib

            with path.open("rb") as f:
                loaded = tomllib.load(f)
            assert loaded["grid"]["file_2d"] == './path/with "quotes" and \\ backslash.vts'
        finally:
            if path.exists():
                path.unlink()
