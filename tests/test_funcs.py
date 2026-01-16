"""Test cases for the Settings and Helpers modules."""

import pathlib
import time
from os.path import join
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import vtk
from numpy.testing import assert_allclose, assert_equal, assert_string_equal
from vtk.util import numpy_support

from fluvial_particle.Helpers import (
    SETTINGS_TEMPLATE,
    convert_grid_hdf5tovtk,
    convert_particles_hdf5tocsv,
    create_parser,
    generate_settings_template,
    get_prng,
    load_checkpoint,
    load_variable_source,
)
from fluvial_particle.RiverGrid import RiverGrid
from fluvial_particle.Settings import Settings

from .support import get_h5file, get_num_timesteps, get_points


def test_conversions(request):
    """Test the HDF5 conversion methods from the Helpers module."""
    with TemporaryDirectory() as tmpdirname:
        tests_dir = request.fspath.dirname
        parts_file = join(tests_dir, "data", "output_straight", "particles.h5")
        cells_file = join(tests_dir, "data", "output_straight", "cells.h5")
        convert_particles_hdf5tocsv(parts_file, tmpdirname)
        convert_grid_hdf5tovtk(cells_file, tmpdirname)

        # Test on particles csv conversion
        parts_out_name = join(tmpdirname, "particles0.csv")
        parts_out = np.loadtxt(parts_out_name, delimiter=",", skiprows=1)
        pts_out = parts_out[:, 1:4]
        test_f = get_h5file(parts_file)
        test_pts = get_points(test_f, 0, twod=False)
        assert_allclose(pts_out, test_pts)

        # Test on mesh vtk conversion
        cellsh5 = get_h5file(cells_file)
        cells_out_name = join(tmpdirname, "cells0.vtk")
        grid = vtk.vtkStructuredGrid()
        reader = vtk.vtkStructuredGridReader()
        reader.SetOutput(grid)
        reader.SetFileName(cells_out_name)
        reader.Update()
        test_x = cellsh5["grid"]["X"][()].ravel()
        out_x = numpy_support.vtk_to_numpy(grid.GetPoints().GetData())[:, 0]
        assert_allclose(test_x, out_x)


def test_create_parser():
    """Test the argparse factory method."""
    parser = create_parser()

    test_1 = [
        "./tests/data/user_options_straight_test.py",
        "./tests/data/output_straight",
    ]
    ns_1 = parser.parse_args(test_1)

    test_2 = [
        "./tests/data/user_options_straight_test.py",
        "./tests/data/output_straight",
        "--seed",
        "10",
        "--no_postprocess",
    ]
    ns_2 = parser.parse_args(test_2)

    assert_string_equal(ns_1.settings_file, "./tests/data/user_options_straight_test.py")
    assert_string_equal(ns_1.output_directory, "./tests/data/output_straight")
    assert_equal(ns_1.no_postprocess, True)
    assert_equal(ns_1.seed, None)

    assert_equal(ns_2.seed, 10)
    assert_equal(ns_2.no_postprocess, False)


def test_create_parser_init_flag():
    """Test that --init flag is recognized."""
    parser = create_parser()

    args = parser.parse_args(["--init"])
    assert_equal(args.init, True)
    assert args.settings_file is None
    assert args.output_directory is None


def test_create_parser_format_flag():
    """Test that --format flag works with --init."""
    parser = create_parser()

    # Default format is toml
    args = parser.parse_args(["--init"])
    assert_equal(args.format, "toml")

    # Explicit toml format
    args = parser.parse_args(["--init", "--format", "toml"])
    assert_equal(args.format, "toml")

    # Python format
    args = parser.parse_args(["--init", "--format", "python"])
    assert_equal(args.format, "python")


def test_create_parser_version_flag():
    """Test that --version flag is recognized."""
    parser = create_parser()

    # --version raises SystemExit, so we need to catch it
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0


def test_generate_settings_template():
    """Test the settings template generator with Python format."""
    with TemporaryDirectory() as tmpdir:
        output_path = join(tmpdir, "test_options.py")
        generate_settings_template(output_path, format="python")

        # Verify file was created
        assert pathlib.Path(output_path).exists()

        # Verify contents (Python format)
        content = pathlib.Path(output_path).read_text(encoding="utf-8")
        assert "field_map_2d" in content
        assert "field_map_3d" in content
        assert "file_name_2d" in content
        assert "SimTime" in content
        assert "ParticleType" in content


def test_generate_settings_template_toml():
    """Test the settings template generator with TOML format (default)."""
    with TemporaryDirectory() as tmpdir:
        output_path = join(tmpdir, "test_options.toml")
        generate_settings_template(output_path)  # Default is TOML

        # Verify file was created
        assert pathlib.Path(output_path).exists()

        # Verify contents (TOML format)
        content = pathlib.Path(output_path).read_text(encoding="utf-8")
        assert "[simulation]" in content
        assert "[particles]" in content
        assert "[grid]" in content
        assert "field_map_2d" in content
        assert 'type = "Particles"' in content


def test_generate_settings_template_no_overwrite():
    """Test that template generator refuses to overwrite existing files."""
    with TemporaryDirectory() as tmpdir:
        output_path = join(tmpdir, "existing.toml")

        # Create existing file
        pathlib.Path(output_path).write_text("existing content", encoding="utf-8")

        # Should raise SystemExit when trying to overwrite
        with pytest.raises(SystemExit) as exc_info:
            generate_settings_template(output_path)
        assert exc_info.value.code == 1


def test_settings_template_content():
    """Test that the settings template contains all required sections."""
    # Check required fields are documented
    assert "field_map_2d" in SETTINGS_TEMPLATE
    assert "field_map_3d" in SETTINGS_TEMPLATE
    assert "file_name_2d" in SETTINGS_TEMPLATE
    assert "file_name_3d" in SETTINGS_TEMPLATE
    assert "SimTime" in SETTINGS_TEMPLATE
    assert "dt" in SETTINGS_TEMPLATE
    assert "PrintAtTick" in SETTINGS_TEMPLATE
    assert "Track3D" in SETTINGS_TEMPLATE
    assert "NumPart" in SETTINGS_TEMPLATE
    assert "StartLoc" in SETTINGS_TEMPLATE
    assert "ParticleType" in SETTINGS_TEMPLATE

    # Check optional fields are documented
    assert "time_dependent" in SETTINGS_TEMPLATE
    assert "grid_interpolation" in SETTINGS_TEMPLATE
    assert "output_vtp" in SETTINGS_TEMPLATE


def test_get_prng():
    """Test the random number generator factory method."""
    tobj = time.time

    prng_1 = get_prng(tobj)

    assert_equal(type(prng_1), np.random.RandomState)


def test_load_checkpoint():
    """Test the function that loads simulation checkpoint data from an existing HDF5 file."""
    fname = "./tests/data/output_straight/particles.h5"
    tidx = -1
    start = 0
    end = 20
    x, y, z, t = load_checkpoint(fname, tidx, start, end)

    assert_equal(t, 60.0)
    assert_equal(type(y), np.ndarray)
    assert_equal(x.size, 20)
    assert_equal(z.ndim, 1)
    assert_equal(x[15], 43.956639705165685)
    assert_equal(y[3], -0.7139319644180326)
    assert_equal(z[2], 9.563715357675903)


def test_load_variable_source():
    """Test the method that loads particles starting location and activation times."""
    fname = "./tests/data/varsrc_straight.csv"
    pstime, x, y, z = load_variable_source(fname)

    assert_equal(type(y), np.ndarray)
    assert_equal(z.size, 20)
    assert_equal(pstime.size, x.size)
    assert_equal(x[-1], 6.14)
    assert_equal(y[0], y[9])
    assert_equal(z[5], 9.5)
    assert_equal(z[16], 9.5)
    assert_equal(pstime[12], 60.0)


def test_settings_module():
    """Test the Settings module."""
    settings_file = "./tests/data/user_options_straight_test.py"
    options = Settings.read(settings_file)

    assert_equal(options["dt"], 0.25)
    assert_equal(options["NumPart"], 20)
    assert_string_equal(options["file_name_2d"], "./tests/data/Result_straight_2d_1.vtk")


def test_support():
    """Test the testing support functions."""
    h5fname = "./tests/data/output_straight/particles.h5"
    test_file = get_h5file(h5fname)
    test_keys = list(test_file.keys())
    test_nts = get_num_timesteps(test_file)
    test_points = get_points(test_file, test_nts - 1)
    test_file.close()

    assert_equal(test_keys, ["coordinates", "properties"])
    assert_equal(test_nts, 4)
    assert_equal(test_points[-1, :], [34.11210153322322, 0.3878199353097108, 9.73657708285166])


def test_field_mapping_required():
    """Test that field_map_2d and field_map_3d are required parameters."""
    # Test missing field_map_2d
    with pytest.raises(ValueError, match="field_map_2d is required"):
        RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d=None,
            field_map_3d={"velocity": "Velocity"},
        )

    # Test missing field_map_3d
    with pytest.raises(ValueError, match="field_map_3d is required"):
        RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "wet_dry": "IBC",
                "shear_stress": "ShearStress (magnitude)",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d=None,
        )


def test_field_mapping_missing_keys():
    """Test that missing required keys in field mappings raise ValueError."""
    # Test missing required key in field_map_2d (wet_dry is optional, but others are required)
    with pytest.raises(ValueError, match="field_map_2d is missing required keys"):
        RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                # Missing required: shear_stress, velocity, water_surface_elevation
            },
            field_map_3d={"velocity": "Velocity"},
        )

    # Test missing key in field_map_3d
    with pytest.raises(ValueError, match="field_map_3d is missing required keys"):
        RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "shear_stress": "ShearStress (magnitude)",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={},  # Missing: velocity
        )


def test_field_mapping_valid():
    """Test that valid field mappings work correctly."""
    # Should not raise any exception
    river = RiverGrid(
        track3d=0,
        filename2d="./tests/data/Result_straight_2d_1.vtk",
        field_map_2d={
            "bed_elevation": "Elevation",
            "wet_dry": "IBC",
            "shear_stress": "ShearStress (magnitude)",
            "velocity": "Velocity",
            "water_surface_elevation": "WaterSurfaceElevation",
        },
        field_map_3d={"velocity": "Velocity"},
    )

    # Verify the grid was loaded correctly
    assert river.vtksgrid2d is not None
    assert river.vtksgrid2d.GetNumberOfPoints() > 0


def test_auto_compute_wet_dry(capsys):
    """Test that wet_dry is computed from depth when not provided in field_map_2d."""
    # Create RiverGrid without wet_dry in field_map_2d
    river = RiverGrid(
        track3d=0,
        filename2d="./tests/data/Result_straight_2d_1.vtk",
        field_map_2d={
            "bed_elevation": "Elevation",
            "shear_stress": "ShearStress (magnitude)",
            "velocity": "Velocity",
            "water_surface_elevation": "WaterSurfaceElevation",
            # wet_dry intentionally omitted - should be computed from depth
        },
        field_map_3d={"velocity": "Velocity"},
        min_depth=0.02,
    )

    # Verify the grid was loaded correctly
    assert river.vtksgrid2d is not None
    assert river.vtksgrid2d.GetNumberOfPoints() > 0

    # Verify that wet_dry was computed (check stdout message)
    captured = capsys.readouterr()
    assert "Computed wet_dry from depth" in captured.out

    # Verify that _compute_wet_dry flag was set
    assert river._compute_wet_dry is True


def test_auto_compute_wet_dry_custom_min_depth(capsys):
    """Test that custom min_depth is used when computing wet_dry."""
    custom_min_depth = 0.05

    river = RiverGrid(
        track3d=0,
        filename2d="./tests/data/Result_straight_2d_1.vtk",
        field_map_2d={
            "bed_elevation": "Elevation",
            "shear_stress": "ShearStress (magnitude)",
            "velocity": "Velocity",
            "water_surface_elevation": "WaterSurfaceElevation",
        },
        field_map_3d={"velocity": "Velocity"},
        min_depth=custom_min_depth,
    )

    # Verify the custom min_depth was used
    captured = capsys.readouterr()
    assert f"min_depth={custom_min_depth}m" in captured.out
    assert river._min_depth == custom_min_depth
