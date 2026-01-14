"""Test cases for time-varying grid functionality."""

import time
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from fluvial_particle import simulate
from fluvial_particle.grids import TimeVaryingGrid
from fluvial_particle.Settings import Settings


class TestTimeVaryingGrid:
    """Tests for TimeVaryingGrid class."""

    def test_time_varying_grid_initialization(self):
        """Test that TimeVaryingGrid initializes correctly."""
        grid = TimeVaryingGrid(
            track3d=1,
            file_pattern_2d="./tests/data/time_series_straight/Result_2D_{}.vts",
            file_pattern_3d="./tests/data/time_series_straight/Result_3D_{}.vts",
            field_map_2d={
                "bed_elevation": "Elevation[m]",
                "shear_stress": "Tausta",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurf.[m]",
            },
            field_map_3d={"velocity": "Velocity"},
            grid_start_index=2,
            grid_end_index=6,
            grid_dt=1.0,
            grid_start_time=0.0,
            interpolation="linear",
        )

        assert len(grid.grid_times) == 5
        assert grid.grid_times[0] == 0.0
        assert grid.grid_times[-1] == 4.0
        assert grid.interpolation == "linear"

    def test_time_varying_grid_advance_to_time(self):
        """Test that advance_to_time loads correct grids."""
        grid = TimeVaryingGrid(
            track3d=0,
            file_pattern_2d="./tests/data/time_series_straight/Result_2D_{}.vts",
            file_pattern_3d="./tests/data/time_series_straight/Result_3D_{}.vts",
            field_map_2d={
                "bed_elevation": "Elevation[m]",
                "shear_stress": "Tausta",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurf.[m]",
            },
            field_map_3d={"velocity": "Velocity"},
            grid_start_index=2,
            grid_end_index=6,
            grid_dt=1.0,
        )

        # Initially at t=0
        assert grid.current_grid_time == 0.0

        # Advance to t=0.5 (should stay on first grid pair)
        updated = grid.advance_to_time(0.5)
        assert not updated
        assert grid.current_grid_time == 0.0

        # Advance to t=1.5 (should load new grid)
        updated = grid.advance_to_time(1.5)
        assert updated
        assert grid.current_grid_time == 1.0

    def test_time_varying_grid_interpolation_weight(self):
        """Test interpolation weight calculation."""
        grid = TimeVaryingGrid(
            track3d=0,
            file_pattern_2d="./tests/data/time_series_straight/Result_2D_{}.vts",
            file_pattern_3d="./tests/data/time_series_straight/Result_3D_{}.vts",
            field_map_2d={
                "bed_elevation": "Elevation[m]",
                "shear_stress": "Tausta",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurf.[m]",
            },
            field_map_3d={"velocity": "Velocity"},
            grid_start_index=2,
            grid_end_index=6,
            grid_dt=1.0,
            interpolation="linear",
        )

        # At t=0, weight should be 0
        grid.advance_to_time(0.0)
        assert grid.get_interpolation_weight() == 0.0

        # At t=0.5, weight should be 0.5
        grid.advance_to_time(0.5)
        assert abs(grid.get_interpolation_weight() - 0.5) < 1e-10

        # At t=0.75, weight should be 0.75
        grid.advance_to_time(0.75)
        assert abs(grid.get_interpolation_weight() - 0.75) < 1e-10

    def test_time_varying_grid_hold_interpolation(self):
        """Test hold interpolation mode (weight always 0)."""
        grid = TimeVaryingGrid(
            track3d=0,
            file_pattern_2d="./tests/data/time_series_straight/Result_2D_{}.vts",
            file_pattern_3d="./tests/data/time_series_straight/Result_3D_{}.vts",
            field_map_2d={
                "bed_elevation": "Elevation[m]",
                "shear_stress": "Tausta",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurf.[m]",
            },
            field_map_3d={"velocity": "Velocity"},
            grid_start_index=2,
            grid_end_index=6,
            grid_dt=1.0,
            interpolation="hold",
        )

        grid.advance_to_time(0.5)
        assert grid.get_interpolation_weight() == 0.0

        grid.advance_to_time(0.9)
        assert grid.get_interpolation_weight() == 0.0

    def test_time_varying_grid_nearest_interpolation(self):
        """Test nearest interpolation mode."""
        grid = TimeVaryingGrid(
            track3d=0,
            file_pattern_2d="./tests/data/time_series_straight/Result_2D_{}.vts",
            file_pattern_3d="./tests/data/time_series_straight/Result_3D_{}.vts",
            field_map_2d={
                "bed_elevation": "Elevation[m]",
                "shear_stress": "Tausta",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurf.[m]",
            },
            field_map_3d={"velocity": "Velocity"},
            grid_start_index=2,
            grid_end_index=6,
            grid_dt=1.0,
            interpolation="nearest",
        )

        # Before midpoint, use current (weight=0)
        grid.advance_to_time(0.4)
        assert grid.get_interpolation_weight() == 0.0

        # After midpoint, use next (weight=1)
        grid.advance_to_time(0.6)
        assert grid.get_interpolation_weight() == 1.0

    def test_time_varying_grid_out_of_range_error(self):
        """Test that out-of-range times raise ValueError."""
        grid = TimeVaryingGrid(
            track3d=0,
            file_pattern_2d="./tests/data/time_series_straight/Result_2D_{}.vts",
            file_pattern_3d="./tests/data/time_series_straight/Result_3D_{}.vts",
            field_map_2d={
                "bed_elevation": "Elevation[m]",
                "shear_stress": "Tausta",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurf.[m]",
            },
            field_map_3d={"velocity": "Velocity"},
            grid_start_index=2,
            grid_end_index=6,
            grid_dt=1.0,
        )

        with pytest.raises(ValueError, match="before first grid time"):
            grid.advance_to_time(-1.0)

        with pytest.raises(ValueError, match="after last grid time"):
            grid.advance_to_time(10.0)

    def test_time_varying_grid_missing_files_error(self):
        """Test that missing grid files raise ValueError."""
        with pytest.raises(ValueError, match="Missing 2D grid files"):
            TimeVaryingGrid(
                track3d=0,
                file_pattern_2d="./tests/data/nonexistent/Result_2D_{}.vts",
                file_pattern_3d="./tests/data/nonexistent/Result_3D_{}.vts",
                field_map_2d={
                    "bed_elevation": "Elevation[m]",
                    "shear_stress": "Tausta",
                    "velocity": "Velocity",
                    "water_surface_elevation": "WaterSurf.[m]",
                },
                field_map_3d={"velocity": "Velocity"},
                grid_start_index=2,
                grid_end_index=6,
                grid_dt=1.0,
            )


class TestTimeVaryingSimulation:
    """Integration tests for time-varying simulation."""

    def test_simulation_with_time_varying_grids(self):
        """Test that simulation runs with time-varying grids."""
        with TemporaryDirectory() as tmpdir:
            argdict = {
                "settings_file": "./tests/data/user_options_time_series.py",
                "seed": 3654125,
                "no_postprocess": False,
                "output_directory": tmpdir,
            }

            settings = Settings.read(argdict["settings_file"])
            simulate(settings, argdict, timer=time.time)

            # Check that output files were created
            import pathlib

            output_dir = pathlib.Path(tmpdir)
            assert (output_dir / "particles.h5").exists()
            assert (output_dir / "particles.xmf").exists()

    def test_simulation_time_varying_produces_valid_output(self):
        """Test that time-varying simulation produces valid particle positions."""
        with TemporaryDirectory() as tmpdir:
            argdict = {
                "settings_file": "./tests/data/user_options_time_series.py",
                "seed": 3654125,
                "no_postprocess": True,
                "output_directory": tmpdir,
            }

            settings = Settings.read(argdict["settings_file"])
            simulate(settings, argdict, timer=time.time)

            # Load and check output
            import h5py

            with h5py.File(f"{tmpdir}/particles.h5", "r") as f:
                x = f["coordinates"]["x"][:]
                y = f["coordinates"]["y"][:]

                # Check that positions are not all NaN
                assert not np.all(np.isnan(x))
                assert not np.all(np.isnan(y))

                # Check that particles moved (final positions differ from initial)
                x_initial = x[0, :]
                x_final = x[-1, :]
                assert not np.allclose(x_initial, x_final, equal_nan=True)
