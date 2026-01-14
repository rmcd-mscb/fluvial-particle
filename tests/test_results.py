"""Test cases for the results module (notebook API)."""

import pathlib
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from numpy.testing import assert_equal

from fluvial_particle.results import SimulationResults, run_simulation


class TestSimulationResults:
    """Tests for the SimulationResults class."""

    @pytest.fixture
    def results(self) -> SimulationResults:
        """Load test simulation results."""
        return SimulationResults("./tests/data/output_straight")

    def test_load_results(self, results):
        """Test loading simulation results."""
        assert results.num_timesteps > 0
        assert results.num_particles > 0

    def test_num_timesteps(self, results):
        """Test num_timesteps property."""
        assert_equal(results.num_timesteps, 4)

    def test_num_particles(self, results):
        """Test num_particles property."""
        assert_equal(results.num_particles, 20)

    def test_times(self, results):
        """Test times property."""
        times = results.times
        assert len(times) == results.num_timesteps
        assert times[0] == 0.0
        assert times[-1] == 60.0

    def test_coordinate_names(self, results):
        """Test coordinate_names property."""
        names = results.coordinate_names
        assert "x" in names
        assert "y" in names
        assert "z" in names
        assert "time" in names

    def test_property_names(self, results):
        """Test property_names property."""
        names = results.property_names
        assert "depth" in names
        assert "velvec" in names
        assert "wse" in names

    def test_get_positions_single_timestep(self, results):
        """Test getting positions for a single timestep."""
        positions = results.get_positions(timestep=0)
        assert positions.shape == (results.num_particles, 3)

    def test_get_positions_all_timesteps(self, results):
        """Test getting positions for all timesteps."""
        positions = results.get_positions()
        assert positions.shape == (results.num_timesteps, results.num_particles, 3)

    def test_get_positions_negative_index(self, results):
        """Test getting positions with negative index."""
        positions = results.get_positions(timestep=-1)
        assert positions.shape == (results.num_particles, 3)

    def test_get_positions_flatten_z(self, results):
        """Test flattening z coordinates."""
        positions = results.get_positions(timestep=0, flatten_z=True)
        assert np.all(positions[:, 2] == 0)

    def test_get_positions_2d_single_timestep(self, results):
        """Test getting 2D positions for a single timestep."""
        positions = results.get_positions_2d(timestep=0)
        assert positions.shape == (results.num_particles, 2)

    def test_get_positions_2d_all_timesteps(self, results):
        """Test getting 2D positions for all timesteps."""
        positions = results.get_positions_2d()
        assert positions.shape == (results.num_timesteps, results.num_particles, 2)

    def test_get_property(self, results):
        """Test getting a property."""
        depths = results.get_property("depth", timestep=0)
        assert depths.shape == (results.num_particles,)

    def test_get_property_all_timesteps(self, results):
        """Test getting property for all timesteps."""
        depths = results.get_property("depth")
        assert depths.shape == (results.num_timesteps, results.num_particles)

    def test_get_property_invalid_name(self, results):
        """Test getting an invalid property raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            results.get_property("nonexistent")

    def test_get_velocities(self, results):
        """Test get_velocities convenience method."""
        velocities = results.get_velocities(timestep=0)
        assert velocities.shape == (results.num_particles, 3)

    def test_get_depths(self, results):
        """Test get_depths convenience method."""
        depths = results.get_depths(timestep=0)
        assert depths.shape == (results.num_particles,)

    def test_summary(self, results):
        """Test summary method."""
        summary = results.summary()
        assert "Timesteps: 4" in summary
        assert "Particles: 20" in summary
        assert "output_straight" in summary

    def test_context_manager(self):
        """Test using as context manager."""
        with SimulationResults("./tests/data/output_straight") as results:
            assert results.num_timesteps == 4

    def test_repr(self, results):
        """Test string representation."""
        repr_str = repr(results)
        assert "SimulationResults" in repr_str
        assert "timesteps=4" in repr_str
        assert "particles=20" in repr_str

    def test_file_not_found(self):
        """Test that missing output raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match=r"particles\.h5 not found"):
            SimulationResults("./nonexistent_directory")


class TestRunSimulation:
    """Tests for the run_simulation convenience function."""

    def test_run_simulation_basic(self):
        """Test running a basic simulation."""
        with TemporaryDirectory() as tmpdir:
            results = run_simulation(
                "./tests/data/user_options_straight_test.py",
                tmpdir,
                seed=12345,
                postprocess=False,
                quiet=True,
            )

            assert results.num_timesteps > 0
            assert results.num_particles == 20
            assert pathlib.Path(tmpdir, "particles.h5").exists()

    def test_run_simulation_creates_output_dir(self):
        """Test that run_simulation creates output directory."""
        with TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir) / "new_output"
            assert not output_dir.exists()

            results = run_simulation(
                "./tests/data/user_options_straight_test.py",
                output_dir,
                seed=12345,
                postprocess=False,
                quiet=True,
            )

            assert output_dir.exists()
            assert results.num_particles == 20

    def test_run_simulation_file_not_found(self):
        """Test that missing settings file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Settings file not found"):
            run_simulation("./nonexistent_settings.py", "./output")

    def test_run_simulation_reproducible(self):
        """Test that simulations with same seed are reproducible."""
        with TemporaryDirectory() as tmpdir1, TemporaryDirectory() as tmpdir2:
            results1 = run_simulation(
                "./tests/data/user_options_straight_test.py",
                tmpdir1,
                seed=42,
                postprocess=False,
                quiet=True,
            )

            results2 = run_simulation(
                "./tests/data/user_options_straight_test.py",
                tmpdir2,
                seed=42,
                postprocess=False,
                quiet=True,
            )

            positions1 = results1.get_positions(timestep=-1)
            positions2 = results2.get_positions(timestep=-1)

            np.testing.assert_array_equal(positions1, positions2)


class TestSimulationResultsDataFrame:
    """Tests for DataFrame export (requires pandas)."""

    @pytest.fixture
    def results(self) -> SimulationResults:
        """Load test simulation results."""
        return SimulationResults("./tests/data/output_straight")

    def test_to_dataframe_single_timestep(self, results):
        """Test converting single timestep to DataFrame."""
        pytest.importorskip("pandas")
        df = results.to_dataframe(timestep=0)

        assert len(df) == results.num_particles
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns
        assert "time" in df.columns
        assert "depth" in df.columns

    def test_to_dataframe_all_timesteps(self, results):
        """Test converting all timesteps to DataFrame."""
        pytest.importorskip("pandas")
        df = results.to_dataframe()

        expected_rows = results.num_timesteps * results.num_particles
        assert len(df) == expected_rows
        assert "timestep" in df.columns

    def test_to_dataframe_velocity_columns(self, results):
        """Test that velocity is split into vx, vy, vz columns."""
        pytest.importorskip("pandas")
        df = results.to_dataframe(timestep=0)

        assert "vx" in df.columns
        assert "vy" in df.columns
        assert "vz" in df.columns
