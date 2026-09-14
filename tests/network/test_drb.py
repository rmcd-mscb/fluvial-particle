"""End-to-end run on the DRB export when FLUVIAL_PARTICLE_DRB_FILE is set."""

import os
import pathlib

import numpy as np
import pytest

from fluvial_particle.network import FileHydraulicsProvider, Network, run_network_simulation


DRB = os.environ.get("FLUVIAL_PARTICLE_DRB_FILE")
pytestmark = pytest.mark.skipif(not DRB or not pathlib.Path(DRB).exists(), reason="DRB file not available")


def test_drb_five_days_from_headwaters(tmp_path):
    with FileHydraulicsProvider(DRB) as prov:
        net = Network(prov.static)
        heads = net.headwaters()
        start = prov.times[0]
    sources = [{"reach_id": int(r), "form": "slug", "time": 0.0, "mass": 10.0} for r in heads]
    sources.append({"reach_id": 4205, "form": "slug", "time": 0.0, "mass": 10.0})  # guarantees an exit at Trenton
    cfg = {
        "hydraulics_file": DRB,
        "dt": 900.0,
        "output_interval": 3600.0,
        "start_time": str(start),
        "end_time": str(start + np.timedelta64(5, "D")),
        "particle_mass": 1.0,
        "sources": sources,
    }
    with run_network_simulation(cfg, tmp_path / "drb", seed=1, quiet=True) as res:
        assert res.n_particles == 10 * (heads.size + 1)
        status = res.positions(-1)["status"]
        assert status.isin([1, 2]).all()
        assert len(res.arrival_times(outlet=4205)) > 0
        conc = res.reach_concentration(-1)
        assert np.nanmax(conc.values) > 0
