"""Run a network simulation under MPI: python mpi_run.py <hydraulics.nc> <output_dir>."""

import sys

from mpi4py import MPI

from fluvial_particle.network.run import run_network_simulation


def main() -> None:
    path, out = sys.argv[1], sys.argv[2]
    cfg = {
        "hydraulics_file": path,
        "dt": 600.0,
        "output_interval": 600.0,
        "end_time": "1979-01-01T02:00",
        "dispersion": {"model": "none"},
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 6.0, "particles": 6},
            {"reach_id": 102, "form": "loading", "rate": 0.001, "start": 0.0, "end": 3600.0, "particles": 5},
        ],
    }
    run_network_simulation(cfg, out, seed=1, comm=MPI.COMM_WORLD, quiet=True)


if __name__ == "__main__":
    main()
