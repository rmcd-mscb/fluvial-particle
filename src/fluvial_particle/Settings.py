"""Settings file, subclass of dictionary."""

import pathlib

from .FallingParticles import FallingParticles
from .LarvalParticles import LarvalBotParticles, LarvalTopParticles
from .Particles import Particles


# New Particles subclasses must be added to global_dict before they can be used
global_dict = {
    "Particles": Particles,
    "FallingParticles": FallingParticles,
    "LarvalTopParticles": LarvalTopParticles,
    "LarvalBotParticles": LarvalBotParticles,
}


class Settings(dict):
    """Handler class to user defined parameters. Allows us to check a users input parameters in the backend."""

    def __init__(self, **kwargs):
        """Check that options file has all required keys."""
        missing = [x for x in self.required_keys if x not in kwargs]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} from the user parameter file")

        for key, value in kwargs.items():
            self[key] = value

    @property
    def required_keys(self):
        """Attributes required in the options file.

        Returns:
            list(str): list of the required keys: SimTime, dt, Track3D, PrintAtTick, file_name_2d, file_name_3d,
            NumPart, StartLoc
        """
        return (
            "SimTime",
            "dt",
            "Track3D",
            "PrintAtTick",
            "file_name_2d",
            "file_name_3d",
            "NumPart",
            "StartLoc",
        )

    @classmethod
    def read(cls, filename):
        """Load user parameters from options file.

        Args:
            filename (str): path to the options file

        Returns:
            Settings: dict-like object with user parameters stored as key: value pairs
        """
        options = {}
        # Load user parameters
        with pathlib.Path(filename).open(encoding="utf-8") as f:
            f = "\n".join(f.readlines())
            exec(f, global_dict, options)  # noqa: S102 # nosec B102 - Intentional exec for user config loading

        return cls(**options)
