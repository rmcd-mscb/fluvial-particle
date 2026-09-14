"""Command-line interface entry points for fluvial particle tracking."""

import argparse
import pathlib
import time
from collections.abc import Sequence

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


def network_parser() -> argparse.ArgumentParser:
    """Argument parser for the network solver entry points.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="fluvial_particle_network",
        description="1D river-network particle tracking from a network hydraulics export.",
    )
    parser.add_argument("settings_file", nargs="?", help="TOML settings file with a [network] table")
    parser.add_argument("-o", "--output", help="output directory (created if missing)")
    parser.add_argument("--seed", type=int, default=None, help="base random seed")
    parser.add_argument("--quiet", action="store_true", help="suppress the startup report")
    parser.add_argument("--init", action="store_true", help="print a settings template and exit")
    return parser


def _network_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse and validate network CLI arguments, handling ``--init``.

    Args:
        argv: command-line arguments, or None to use sys.argv.

    Returns:
        The parsed and validated namespace.

    Raises:
        FileNotFoundError: if the settings file does not exist.
    """
    from .network.config import get_network_config_template

    parser = network_parser()
    args = parser.parse_args(argv)
    if args.init:
        print(get_network_config_template())
        raise SystemExit(0)
    if args.settings_file is None or args.output is None:
        parser.error("settings_file and --output are required (unless using --init)")
    if not pathlib.Path(args.settings_file).exists():
        raise FileNotFoundError(f"Cannot find settings file {args.settings_file}")
    return args


def network_serial(argv: Sequence[str] | None = None) -> None:
    """Run the network solver in serial.

    Args:
        argv: command-line arguments, or None to use sys.argv.
    """
    from .network.run import run_network_simulation

    args = _network_args(argv)
    run_network_simulation(args.settings_file, args.output, seed=args.seed, quiet=args.quiet)


def network_mpi(argv: Sequence[str] | None = None) -> None:
    """Run the network solver under MPI (mpiexec -n N fluvial_particle_network_mpi ...).

    Args:
        argv: command-line arguments, or None to use sys.argv.
    """
    from mpi4py import MPI

    from .network.run import run_network_simulation

    args = _network_args(argv)
    run_network_simulation(args.settings_file, args.output, seed=args.seed, comm=MPI.COMM_WORLD, quiet=args.quiet)
