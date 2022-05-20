"""Test cases for the __main__ module."""
import time

import pytest
from numpy.testing import assert_allclose

from .support import get_h5file
from .support import get_num_timesteps
from .support import get_points
from fluvial_particle import Settings
from fluvial_particle import simulate

# import os

# def test_Particles():
# print(f'the current working directory is {os.getcwd()}')
argdict = {
    "settings_file": "./tests/data/user_options_test.py",
    "output_directory": "./tests/data/output",
    "seed": 3654125,
    "no_postprocess": True,
}


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
                "settings_file": "./tests/data/user_options_test.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_fixed",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_falling.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_falling_fixed",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_larvalbot.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_larvbot_fixed",
        ),
        (
            {
                "settings_file": "./tests/data/user_options_larvaltop.py",
                "output_directory": "./tests/data/output",
                "seed": 3654125,
                "no_postprocess": True,
            },
            "./tests/data/output_larvtop_fixed",
        ),
    ],
)
def test_particle(argdict: dict, test_out_path: str) -> None:
    """Test basic particle-tracking."""
    run_simulation(argdict)
    # get particle output file
    new_file = get_h5file(str(argdict.get("output_directory")) + "/particles.h5")
    new_nts = get_num_timesteps(new_file)
    new_points = get_points(new_file, new_nts - 1, twod=True)
    print(type(new_points), new_points.shape)
    test_file = get_h5file(str(test_out_path) + "/particles.h5")
    test_nts = get_num_timesteps(test_file)
    test_points = get_points(test_file, test_nts - 1, twod=True)

    assert_allclose(test_points, new_points)
