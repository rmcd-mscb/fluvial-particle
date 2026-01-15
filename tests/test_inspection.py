"""Tests for grid inspection utilities."""

import pytest

from fluvial_particle import inspect_grid


class TestInspectGridStatic:
    """Tests for inspect_grid with static grids."""

    def test_inspect_grid_basic(self):
        """Test basic inspect_grid functionality."""
        info = inspect_grid("./tests/data/user_options_straight_test.py", quiet=True)

        # Check structure
        assert "grid_2d" in info
        assert "hydraulics" in info
        assert "time_dependent" in info
        assert "ustar_method" in info

        # Should not be time-dependent
        assert info["time_dependent"] is False

    def test_grid_2d_info(self):
        """Test 2D grid information extraction."""
        info = inspect_grid("./tests/data/user_options_straight_test.py", quiet=True)

        g2d = info["grid_2d"]
        assert "file" in g2d
        assert "dimensions" in g2d
        assert "extents" in g2d
        assert "scalars" in g2d
        assert "vectors" in g2d
        assert "num_points" in g2d
        assert "num_cells" in g2d

        # Check dimensions structure
        dims = g2d["dimensions"]
        assert "i" in dims
        assert "j" in dims
        assert dims["i"] > 0
        assert dims["j"] > 0

        # Check extents structure
        ext = g2d["extents"]
        assert "x" in ext
        assert "y" in ext
        assert "z" in ext
        assert len(ext["x"]) == 2  # (min, max)

    def test_grid_3d_info_when_track3d(self):
        """Test 3D grid info is present when Track3D=1."""
        info = inspect_grid("./tests/data/user_options_straight_test.py", quiet=True)

        # Track3D=1 in test file, so 3D grid should be present
        assert "grid_3d" in info

        g3d = info["grid_3d"]
        dims = g3d["dimensions"]
        assert "i" in dims
        assert "j" in dims
        assert "k" in dims
        assert dims["k"] > 0

    def test_hydraulics_stats(self):
        """Test hydraulic statistics are computed."""
        info = inspect_grid("./tests/data/user_options_straight_test.py", quiet=True)

        hyd = info["hydraulics"]
        assert "depth" in hyd
        assert "velocity_mag" in hyd
        assert "shear_stress" in hyd
        assert "ustar" in hyd

        # Check stat structure
        for key in ["depth", "shear_stress", "ustar"]:
            stats = hyd[key]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats

    def test_ustar_method_detected(self):
        """Test that u* method is correctly detected."""
        info = inspect_grid("./tests/data/user_options_straight_test.py", quiet=True)

        # Test file has shear_stress in field_map_2d
        assert info["ustar_method"] == "shear_stress"

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing settings file."""
        with pytest.raises(FileNotFoundError, match="Settings file not found"):
            inspect_grid("nonexistent_file.py", quiet=True)


class TestInspectGridTimeSeries:
    """Tests for inspect_grid with time-varying grids."""

    def test_time_dependent_basic(self):
        """Test basic time-dependent grid inspection."""
        info = inspect_grid("./tests/data/user_options_time_series.py", quiet=True)

        # Should be time-dependent
        assert isinstance(info["time_dependent"], dict)
        assert info["time_dependent"]["enabled"] is True

    def test_time_dependent_info_structure(self):
        """Test time-dependent info structure."""
        info = inspect_grid("./tests/data/user_options_time_series.py", quiet=True)

        td = info["time_dependent"]
        assert "timestep" in td
        assert "time" in td
        assert "total_timesteps" in td
        assert "grid_dt" in td
        assert "time_range" in td

        # Check values
        assert td["total_timesteps"] == 5  # grid_end_index - grid_start_index + 1 = 6 - 2 + 1 = 5
        assert td["grid_dt"] == 1.0

    def test_timestep_parameter(self):
        """Test that timestep parameter selects correct grid."""
        # Default (timestep=0)
        info0 = inspect_grid("./tests/data/user_options_time_series.py", timestep=0, quiet=True)
        assert info0["time_dependent"]["timestep"] == 0
        assert info0["time_dependent"]["time"] == 0.0

        # Timestep 2
        info2 = inspect_grid("./tests/data/user_options_time_series.py", timestep=2, quiet=True)
        assert info2["time_dependent"]["timestep"] == 2
        assert info2["time_dependent"]["time"] == 2.0

    def test_negative_timestep(self):
        """Test that negative timestep works like Python indexing."""
        info = inspect_grid("./tests/data/user_options_time_series.py", timestep=-1, quiet=True)

        # -1 should be last timestep (4)
        assert info["time_dependent"]["timestep"] == 4
        assert info["time_dependent"]["time"] == 4.0

    def test_timestep_out_of_range(self):
        """Test that out-of-range timestep raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            inspect_grid("./tests/data/user_options_time_series.py", timestep=10, quiet=True)


class TestInspectGridOutput:
    """Tests for inspect_grid output formatting."""

    def test_quiet_mode(self, capsys):
        """Test that quiet mode suppresses output."""
        inspect_grid("./tests/data/user_options_straight_test.py", quiet=True)
        captured = capsys.readouterr()

        # Should only have the u* method print from RiverGrid initialization
        # The formatted summary should NOT be printed
        assert "Grid Summary" not in captured.out

    def test_verbose_mode(self, capsys):
        """Test that verbose mode prints summary."""
        inspect_grid("./tests/data/user_options_straight_test.py", quiet=False)
        captured = capsys.readouterr()

        assert "Grid Summary" in captured.out
        assert "2D Grid:" in captured.out
        assert "Dimensions" in captured.out
        assert "Extents" in captured.out
        assert "Reach-Averaged Hydraulics" in captured.out


class TestHydraulicsComputation:
    """Tests for hydraulic statistics computation."""

    def test_depth_statistics(self):
        """Test that depth statistics are reasonable."""
        info = inspect_grid("./tests/data/user_options_straight_test.py", quiet=True)

        depth = info["hydraulics"]["depth"]
        # Depth should be positive
        assert depth["min"] >= 0
        assert depth["mean"] > 0
        assert depth["max"] > depth["min"]
        # Std should be non-negative
        assert depth["std"] >= 0

    def test_ustar_statistics(self):
        """Test that u* statistics are reasonable."""
        info = inspect_grid("./tests/data/user_options_straight_test.py", quiet=True)

        ustar = info["hydraulics"]["ustar"]
        # u* should be non-negative
        assert ustar["min"] >= 0
        assert ustar["mean"] >= 0
        # Should have some spread
        assert ustar["max"] >= ustar["mean"]
