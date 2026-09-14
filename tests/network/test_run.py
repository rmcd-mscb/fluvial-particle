"""End-to-end tests for run_network_simulation."""

import numpy as np
import pytest
import xarray as xr

from fluvial_particle.network.run import resolve_seed, run_network_simulation
from fluvial_particle.network.writer import OUTPUT_FILENAME
from tests.network.support import three_reach_dataset, write_network_file


def config_for(path, **extra):
    cfg = {
        "hydraulics_file": str(path),
        "dt": 600.0,
        "output_interval": 600.0,
        "end_time": "1979-01-01T02:00",
        "dispersion": {"model": "none"},
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 3.0, "particles": 3},
            {"reach_id": 102, "form": "loading", "rate": 0.001, "start": 0.0, "end": 3600.0, "particles": 2},
        ],
    }
    cfg.update(extra)
    return cfg


def test_run_writes_file_and_conserves_mass(tmp_path):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    res = run_network_simulation(config_for(path), tmp_path / "out", seed=1, quiet=True)
    assert (tmp_path / "out" / OUTPUT_FILENAME).exists()
    assert res.n_particles == 5
    assert res.times.size == 13  # t = 0 plus 12 hourly-tenth outputs over 2 h at 600 s
    with xr.open_dataset(tmp_path / "out" / OUTPUT_FILENAME, engine="h5netcdf") as ds:
        status = ds["status"].values
        assert all(np.bincount(row, minlength=3).sum() == 5 for row in status)  # every particle accounted for
        # slug exited (3); of the loading source's 2 particles (released at t=900s and t=2700s, each
        # needing 5500 s to reach the outlet), the first exits at t=6400s (before the 7200s run end)
        # and the second (exit at t=8200s) is still moving.
        assert list(np.bincount(status[-1], minlength=3)) == [0, 1, 4]
        # reach 101 particles: 1000 m at 1 m/s then 3000 m at 2 m/s = 2500 s
        np.testing.assert_allclose(ds["exit_time"].values[:3], 2500.0)
        assert list(ds["exit_reach"].values[:3]) == [2, 2, 2]
        assert ds.attrs["seed"] == 1 and ds.attrs["interpolation"] == "linear"
        assert ds.attrs["hydraulics_file"].endswith("net.nc")
        assert "sources" in ds.attrs and "dispersion" in ds.attrs
        assert ds["reach_index"].values[1, 0] == 0 and ds["s"].values[1, 0] == 600.0
    res.close()


def test_run_accepts_toml_and_prints_report(tmp_path, capsys):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    toml = tmp_path / "run.toml"
    lines = [
        "[network]",
        f'hydraulics_file = "{path}"',
        "dt = 600.0",
        "output_interval = 1200.0",
        'end_time = "1979-01-01T01:00"',
        "particle_mass = 1.0",
        "[network.dispersion]",
        'model = "none"',
        "[[network.sources]]",
        "reach_id = 101",
        'form = "slug"',
        "time = 0.0",
        "mass = 2.0",
    ]
    toml.write_text("\n".join(lines) + "\n")
    res = run_network_simulation(toml, tmp_path / "out2")
    out = capsys.readouterr().out
    assert "reaches" in out and "dispersion kick" in out and "memory" in out.lower()
    assert "particle mass" in out and "source 0" in out
    assert res.times.size == 4
    res.close()


def test_run_warns_on_non_integer_steps(tmp_path):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    cfg = config_for(path, dt=700.0, output_interval=700.0)
    with pytest.warns(UserWarning, match="stops"):
        res = run_network_simulation(cfg, tmp_path / "out3", seed=1, quiet=True)
    assert res.attrs["end_time"] == "1979-01-01T01:56:40"
    res.close()


def test_resolve_seed():
    assert resolve_seed(5, None) == 5
    s = resolve_seed(None, None)
    assert isinstance(s, int) and s >= 0
