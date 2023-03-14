"""Test cases for the __main__ module."""
import time
from os.path import join
from tempfile import TemporaryDirectory

import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from .support import get_h5file
from .support import get_num_timesteps
from .support import get_points
from fluvial_particle import simulate
from fluvial_particle.Settings import Settings


pytest_plugins = ["pytester"]  # allows testing of command-line applications


def run_simulation(argdict: dict) -> None:
    """Run simulation."""
    settings_file = argdict["settings_file"]
    options = Settings.read(settings_file)
    simulate(options, argdict, timer=time.time)


@pytest.mark.parametrize(
    "argdict, test_out_path",
    [
        (
            {
                "settings_file": "./tests/data/user_options_straight_test.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_straight",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_straight_falling.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_straight_falling",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_straight_larvalbot.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_straight_larvalbot",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_straight_larvaltop.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_straight_larvaltop",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_straight_varsrc.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_straight_varsrc",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_straight_checkpoint.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_straight_checkpoint",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_straight_npz.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_straight",
        ),
    ],
    ids=(
        "Particles simulation",
        "FallingParticles simulation",
        "LarvalBotParticles simulation",
        "LarvalTopParticles simulation",
        "simulate with variable start times",
        "simulate from checkpoint",
        "simulate with npz input meshes",
    ),
)
def test_particle(argdict: dict, test_out_path: str) -> None:
    """Test basic particle-tracking."""
    with TemporaryDirectory() as tmpdirname:
        argdict["output_directory"] = tmpdirname
        run_simulation(argdict)
        # get particle output file
        new_file = get_h5file(str(argdict.get("output_directory")) + "/particles.h5")
        new_nts = get_num_timesteps(new_file)
        new_points = get_points(new_file, new_nts - 1, twod=True)
        test_file = get_h5file(f"{test_out_path}/particles.h5")
        test_nts = get_num_timesteps(test_file)
        test_points = get_points(test_file, test_nts - 1, twod=True)

        assert_allclose(test_points, new_points, atol=1e-4, rtol=0.0)


@pytest.fixture
def run(testdir):
    """Runs fluvial_particle from command line."""

    def do_run(*args):
        args = ["fluvial_particle"] + list(args)
        return testdir.run(*args)

    return do_run


def test_track_serial(run, request, testdir):
    """Test the track_serial command-line entrypoint.

    This test inspired by https://stackoverflow.com/a/13500346
    """
    # First get the paths to the input grids
    grid2d_file = join(
        request.fspath.dirname, "data", "Result_straight_2d_1.vtk"
    )
    grid3d_file = join(
        request.fspath.dirname, "data", "Result_straight_3d_1.vtk"
    )
    # Create user options as list of strings
    arg_list = [
        "from fluvial_particle.Particles import Particles",
        "file_name_2d = " + '"' + grid2d_file + '"',
        "file_name_3d = " + '"' + grid3d_file + '"',
        "SimTime = 60.0",
        "Track3D = 1",
        "dt = 0.25",
        "PrintAtTick = 20.0",
        "NumPart = 20",
        "StartLoc = (5.0, 0.0, 9.5)",
        "startfrac = 0.5",
        "ParticleType = Particles",
        "lev = 0.00025",
    ]
    # Create a new options file in the test directory
    settings_file = join(testdir.tmpdir, "options.py")
    with open(settings_file, "w") as f:
        for line in arg_list:
            f.write(f"{line}\n")
    # Run fluvial_particle from the command-line
    result = run("--seed", "3654125", "--no-postprocess", settings_file, "./")
    # For this test, only assert that fluvial_particle completed successfully
    assert_equal(result.ret, 0)
