"""Tests for NetworkConfig."""

import json
import sys

import numpy as np
import pytest

from fluvial_particle.network.config import DispersionConfig, NetworkConfig, get_network_config_template, parse_datetime


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

MINIMAL = {
    "hydraulics_file": "net.nc",
    "particle_mass": 1.0,
    "sources": [{"reach_id": 101, "form": "slug", "time": 0.0, "mass": 5.0}],
}


def test_defaults_and_parse():
    cfg = NetworkConfig.from_dict(MINIMAL)
    assert cfg.dt == 900.0 and cfg.output_interval == 3600.0
    assert cfg.interpolation == "linear" and cfg.dtype == "float64"
    assert cfg.dispersion == DispersionConfig()
    assert cfg.sources[0]["reach_id"] == 101
    assert cfg.start_time is None


def test_parse_datetime_forms():
    assert parse_datetime("1979-03-01") == np.datetime64("1979-03-01T00:00:00", "ns")
    assert parse_datetime(np.datetime64("1979-03-01T06:00")) == np.datetime64("1979-03-01T06:00", "ns")
    import datetime as dtm

    assert parse_datetime(dtm.datetime(1979, 3, 1, 6)) == np.datetime64("1979-03-01T06:00", "ns")


def test_unknown_key_and_bad_values():
    with pytest.raises(ValueError, match="unknown"):
        NetworkConfig.from_dict({**MINIMAL, "bogus": 1})
    with pytest.raises(ValueError, match="output_interval"):
        NetworkConfig.from_dict({**MINIMAL, "dt": 900.0, "output_interval": 1000.0})
    with pytest.raises(ValueError, match="interpolation"):
        NetworkConfig.from_dict({**MINIMAL, "interpolation": "cubic"})
    with pytest.raises(ValueError, match="sources"):
        NetworkConfig.from_dict({**MINIMAL, "sources": []})
    with pytest.raises(ValueError, match="form"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{"reach_id": 1, "form": "leak"}]})
    with pytest.raises(ValueError, match="particle"):
        NetworkConfig.from_dict({
            "hydraulics_file": "n.nc",
            "sources": [{"reach_id": 1, "form": "slug", "time": 0, "mass": 1}],
        })
    with pytest.raises(ValueError, match="s_frac"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{**MINIMAL["sources"][0], "s": 1.0, "s_frac": 0.5}]})
    with pytest.raises(ValueError, match="dispersion"):
        NetworkConfig.from_dict({**MINIMAL, "dispersion": {"model": "constant"}})
    with pytest.raises(ValueError, match="reach_subset"):
        NetworkConfig.from_dict({**MINIMAL, "reach_subset": {"inlet": 3}})


def test_resolve_times():
    times = np.datetime64("1979-01-01", "ns") + np.arange(5) * np.timedelta64(1, "D")
    cfg = NetworkConfig.from_dict(MINIMAL)
    assert cfg.resolve_times(times) == (times[0], times[-1])
    cfg = NetworkConfig.from_dict({**MINIMAL, "start_time": "1979-01-02", "end_time": "1979-01-03T12:00"})
    assert cfg.resolve_times(times) == (np.datetime64("1979-01-02", "ns"), np.datetime64("1979-01-03T12:00", "ns"))
    with pytest.raises(ValueError, match="end_time"):
        NetworkConfig.from_dict({**MINIMAL, "end_time": "1979-02-01"}).resolve_times(times)
    with pytest.raises(ValueError, match="start_time"):
        NetworkConfig.from_dict({**MINIMAL, "start_time": "1979-01-03", "end_time": "1979-01-02"})


def test_toml_round_trip(tmp_path):
    text = get_network_config_template()
    parsed = tomllib.loads(text)
    assert "network" in parsed and parsed["network"]["sources"]
    path = tmp_path / "s.toml"
    path.write_text(text)
    cfg = NetworkConfig.from_toml(path)
    assert cfg.hydraulics_file.endswith(".nc")
    d = cfg.to_dict()
    json.dumps(d)  # JSON-safe
    assert NetworkConfig.from_dict(d) == cfg
    assert NetworkConfig.coerce(path) == cfg
    assert NetworkConfig.coerce(cfg) is cfg
    (tmp_path / "empty.toml").write_text("x = 1\n")
    with pytest.raises(ValueError, match=r"\[network\]"):
        NetworkConfig.from_toml(tmp_path / "empty.toml")
