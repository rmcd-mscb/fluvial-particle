"""Two-rank MPI write compared with the serial run (dispersion off, so results are identical)."""

import pathlib
import shutil
import subprocess  # noqa: S404 - mpiexec launch under a controlled test path
import sys

import numpy as np
import pytest
import xarray as xr

from fluvial_particle.network.run import run_network_simulation
from tests.network.support import three_reach_dataset, write_network_file


mpi4py = pytest.importorskip("mpi4py")
h5py = pytest.importorskip("h5py")
if not h5py.get_config().mpi:
    pytest.skip("h5py without MPI support", allow_module_level=True)
if shutil.which("mpiexec") is None:
    pytest.skip("mpiexec not found", allow_module_level=True)


def test_two_ranks_match_serial(tmp_path):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    script = pathlib.Path(__file__).with_name("mpi_run.py")
    subprocess.run(  # noqa: S603 - fixed argv, no shell, args controlled by this test
        ["mpiexec", "-n", "2", sys.executable, str(script), str(path), str(tmp_path / "par")],  # noqa: S607
        check=True,
        timeout=300,
    )
    cfg = {
        "hydraulics_file": str(path),
        "dt": 600.0,
        "output_interval": 600.0,
        "end_time": "1979-01-01T02:00",
        "dispersion": {"model": "none"},
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 6.0, "particles": 6},
            {"reach_id": 102, "form": "loading", "rate": 0.001, "start": 0.0, "end": 3600.0, "particles": 5},
        ],
    }
    run_network_simulation(cfg, tmp_path / "serial", seed=1, quiet=True).close()
    with (
        xr.open_dataset(tmp_path / "par" / "network_particles.nc", engine="h5netcdf") as a,
        xr.open_dataset(tmp_path / "serial" / "network_particles.nc", engine="h5netcdf") as b,
    ):
        for name in ("reach_index", "s", "status", "mass", "release_time", "exit_time", "exit_reach"):
            np.testing.assert_array_equal(a[name].values, b[name].values)
