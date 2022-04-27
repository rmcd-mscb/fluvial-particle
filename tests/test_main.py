"""Test cases for the __main__ module."""
import time

from fluvial_particle import Settings
from fluvial_particle import simulate

argdict = {
    "settings_file": "./tests/user_options_rmcd.py",
    "output_directory": "./tests/.",
    "seed": None,
    "no_postprocess": True,
}
settings_file = argdict["settings_file"]
options = Settings.read(settings_file)
simulate(options, argdict, timer=time.time)
