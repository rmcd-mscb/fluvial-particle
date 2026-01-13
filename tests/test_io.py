"""Test cases for the io module (VTP/PVD writers)."""

import pathlib
import time
from tempfile import TemporaryDirectory

import numpy as np
import vtk

from fluvial_particle import simulate
from fluvial_particle.io import PVDWriter, VTPWriter
from fluvial_particle.Settings import Settings


class MockParticles:
    """Mock Particles object for testing VTPWriter."""

    def __init__(self, n_particles: int = 10):
        """Create mock particle data."""
        self.x = np.linspace(0, 10, n_particles)
        self.y = np.linspace(0, 5, n_particles)
        self.z = np.linspace(0, 2, n_particles)
        self.depth = np.full(n_particles, 1.5)
        self.bedelev = np.zeros(n_particles)
        self.wse = np.full(n_particles, 1.5)
        self.htabvbed = np.linspace(0, 1.5, n_particles)
        self.shearstress = np.full(n_particles, 0.5)
        self.cellindex2d = np.arange(n_particles, dtype=np.int64)
        self.cellindex3d = np.arange(n_particles, dtype=np.int64)
        self.velx = np.full(n_particles, 0.5)
        self.vely = np.full(n_particles, 0.1)
        self.velz = np.zeros(n_particles)


class TestVTPWriter:
    """Tests for VTPWriter class."""

    def test_vtp_writer_creates_directory(self):
        """Test that VTPWriter creates output directory."""
        with TemporaryDirectory() as tmpdir:
            vtp_dir = pathlib.Path(tmpdir) / "vtp_output"
            VTPWriter(vtp_dir)
            assert vtp_dir.exists()

    def test_vtp_writer_writes_file(self):
        """Test that VTPWriter writes a VTP file."""
        with TemporaryDirectory() as tmpdir:
            vtp_dir = pathlib.Path(tmpdir) / "vtp"
            writer = VTPWriter(vtp_dir)
            particles = MockParticles(n_particles=10)

            vtp_file = writer.write(particles, time=0.0, tidx=0)

            assert vtp_file is not None
            assert vtp_file.exists()
            assert vtp_file.name == "particles_0000.vtp"

    def test_vtp_writer_returns_none_for_empty_particles(self):
        """Test that VTPWriter returns None when all particles are NaN."""
        with TemporaryDirectory() as tmpdir:
            vtp_dir = pathlib.Path(tmpdir) / "vtp"
            writer = VTPWriter(vtp_dir)
            particles = MockParticles(n_particles=10)
            particles.x = np.full(10, np.nan)

            vtp_file = writer.write(particles, time=0.0, tidx=0)

            assert vtp_file is None

    def test_vtp_file_contains_correct_data(self):
        """Test that VTP file contains correct point and attribute data."""
        with TemporaryDirectory() as tmpdir:
            vtp_dir = pathlib.Path(tmpdir) / "vtp"
            writer = VTPWriter(vtp_dir)
            particles = MockParticles(n_particles=10)

            vtp_file = writer.write(particles, time=5.0, tidx=1)

            # Read the VTP file back
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(str(vtp_file))
            reader.Update()
            polydata = reader.GetOutput()

            # Check number of points
            assert polydata.GetNumberOfPoints() == 10

            # Check point data arrays exist
            point_data = polydata.GetPointData()
            assert point_data.GetArray("Depth") is not None
            assert point_data.GetArray("BedElevation") is not None
            assert point_data.GetArray("Velocity") is not None

            # Check field data contains time
            field_data = polydata.GetFieldData()
            time_arr = field_data.GetArray("TimeValue")
            assert time_arr is not None
            assert time_arr.GetValue(0) == 5.0


class TestPVDWriter:
    """Tests for PVDWriter class."""

    def test_pvd_writer_creates_file(self):
        """Test that PVDWriter creates a PVD file."""
        with TemporaryDirectory() as tmpdir:
            pvd_path = pathlib.Path(tmpdir) / "particles.pvd"
            writer = PVDWriter(pvd_path)
            writer.add_timestep(0.0, pathlib.Path(tmpdir) / "vtp" / "particles_0000.vtp")
            writer.write()

            assert pvd_path.exists()

    def test_pvd_writer_content(self):
        """Test that PVD file contains correct XML structure."""
        with TemporaryDirectory() as tmpdir:
            pvd_path = pathlib.Path(tmpdir) / "particles.pvd"
            writer = PVDWriter(pvd_path)
            writer.add_timestep(0.0, pathlib.Path(tmpdir) / "vtp" / "particles_0000.vtp")
            writer.add_timestep(1.0, pathlib.Path(tmpdir) / "vtp" / "particles_0001.vtp")
            writer.write()

            content = pvd_path.read_text()
            assert '<?xml version="1.0"?>' in content
            assert '<VTKFile type="Collection"' in content
            assert 'timestep="0.0"' in content
            assert 'timestep="1.0"' in content

    def test_pvd_writer_len(self):
        """Test that PVDWriter reports correct number of timesteps."""
        with TemporaryDirectory() as tmpdir:
            pvd_path = pathlib.Path(tmpdir) / "particles.pvd"
            writer = PVDWriter(pvd_path)

            assert len(writer) == 0

            writer.add_timestep(0.0, "dummy.vtp")
            assert len(writer) == 1

            writer.add_timestep(1.0, "dummy2.vtp")
            assert len(writer) == 2


class TestVTPSimulationIntegration:
    """Integration tests for VTP output in simulation."""

    def test_simulation_with_vtp_output(self):
        """Test that simulation creates VTP output when enabled."""
        with TemporaryDirectory() as tmpdir:
            argdict = {
                "settings_file": "./tests/data/user_options_straight_test.py",
                "seed": 3654125,
                "no_postprocess": True,
                "output_directory": tmpdir,
            }

            # Read settings and add output_vtp option
            settings = Settings.read(argdict["settings_file"])
            settings["output_vtp"] = True

            simulate(settings, argdict, timer=time.time)

            # Check VTP directory was created
            vtp_dir = pathlib.Path(tmpdir) / "vtp"
            assert vtp_dir.exists()

            # Check VTP files were created
            vtp_files = list(vtp_dir.glob("particles_*.vtp"))
            assert len(vtp_files) > 0

            # Check PVD file was created
            pvd_file = pathlib.Path(tmpdir) / "particles.pvd"
            assert pvd_file.exists()

            # Verify PVD references the VTP files
            pvd_content = pvd_file.read_text()
            assert "vtp/particles_" in pvd_content
