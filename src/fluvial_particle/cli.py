"""Command-line interface entry points for fluvial particle tracking."""

import time

from .Helpers import checkcommandarguments
from .Settings import Settings
from .simulation import simulate


def track_serial():
    """Run fluvial particle in serial."""
    argdict = checkcommandarguments()
    settings_file = argdict["settings_file"]
    options = Settings.read(settings_file)

    simulate(options, argdict, timer=time.time)


def track_mpi():
    """Run fluvial particle in parallel."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    argdict = checkcommandarguments()
    settings_file = argdict["settings_file"]
    seed = argdict["seed"]
    if seed is not None:
        print("Warning: user-input seed ignored in parallel execution mode.")
        argdict["seed"] = None
    options = Settings.read(settings_file)

    simulate(options, argdict, timer=MPI.Wtime, comm=comm)
