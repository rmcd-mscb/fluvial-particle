"""Fluvial Particle - Lagrangian particle tracking for fluvial environments."""

__author__ = "Richard McDonald"
__email__ = "rmcd@usgs.gov"
__version__ = "0.0.6"

# Re-export public API
from .cli import track_mpi, track_serial
from .inspection import inspect_grid
from .results import SimulationResults, run_simulation
from .simulation import simulate


__all__ = [
    "SimulationResults",
    "inspect_grid",
    "run_simulation",
    "simulate",
    "track_mpi",
    "track_serial",
]
