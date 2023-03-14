"""Test cases for the Settings and Helpers modules."""
import time
from os.path import join
from tempfile import TemporaryDirectory

import numpy as np
import vtk
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_string_equal
from vtk.util import numpy_support

from .support import get_h5file
from .support import get_num_timesteps
from .support import get_points
from fluvial_particle.Helpers import convert_grid_hdf5tovtk
from fluvial_particle.Helpers import convert_particles_hdf5tocsv
from fluvial_particle.Helpers import create_parser
from fluvial_particle.Helpers import get_prng
from fluvial_particle.Helpers import load_checkpoint
from fluvial_particle.Helpers import load_variable_source
from fluvial_particle.Settings import Settings


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

    test_1 = ["./tests/data/user_options_straight_test.py", "./tests/data/output_straight"]
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
    assert_equal(x[15], 43.95664022168961)
    assert_equal(y[3], -0.713931964362969)
    assert_equal(z[2], 9.56371535663352)


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
    assert_string_equal(
        options["file_name_2d"], "./tests/data/Result_straight_2d_1.vtk"
    )


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
    assert_equal(test_points[-1, :], [34.11209947262923, 0.3878199351536123, 9.736577084886918])
