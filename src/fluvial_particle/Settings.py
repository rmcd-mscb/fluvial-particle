"""Settings file, subclass of dictionary."""


class Settings(dict):
    """Handler class to user defined parameters. Allows us to check a users input parameters in the backend."""

    def __init__(self, **kwargs):
        """Check that options file has all required keys."""
        missing = [x for x in self.required_keys if x not in kwargs]
        if len(missing) > 0:
            raise ValueError("Missing {} from the user parameter file".format(missing))

        for key, value in kwargs.items():
            self[key] = value

    @property
    def required_keys(self):
        """Attributes required in the options file."""
        return (
            "SimTime",
            "dt",
            "min_depth",
            "beta_x",
            "beta_y",
            "beta_z",
            "Track3D",
            "PrintAtTick",
            "file_name_2da",
            "file_name_3da",
            "NumPart",
            "StartLoc",
            "amplitude",
            "period",
            "min_elev",
        )

    @classmethod
    def read(cls, filename):
        """Load user parametrs from options file."""
        options = {}
        # Load user parameters
        with open(filename, "r") as f:
            f = "\n".join(f.readlines())
            exec(f, {}, options)  # noqa

        return cls(**options)
