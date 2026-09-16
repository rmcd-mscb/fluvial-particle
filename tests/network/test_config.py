"""Tests for NetworkConfig."""

import json
import sys

import numpy as np
import pytest

from fluvial_particle.network.config import (
    DispersionConfig,
    NetworkConfig,
    ParticlesConfig,
    get_network_config_template,
    parse_datetime,
)


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


def test_source_form_requires_its_keys():
    with pytest.raises(ValueError, match=r"sources\[0\].*slug.*time and mass"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{"reach_id": 1, "form": "slug", "mass": 1.0}]})
    with pytest.raises(ValueError, match=r"sources\[0\].*slug.*time and mass"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{"reach_id": 1, "form": "slug", "time": 0.0}]})
    with pytest.raises(ValueError, match=r"sources\[0\].*loading.*rate or curve"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{"reach_id": 1, "form": "loading", "particles": 1}]})
    with pytest.raises(ValueError, match=r"sources\[0\].*concentration.*value or curve"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{"reach_id": 1, "form": "concentration", "particles": 1}]})


def test_toml_source_with_unquoted_datetime_is_json_safe(tmp_path):
    # tomllib parses an unquoted TOML datetime (no surrounding quotes) as datetime.datetime, not str;
    # to_dict() (and thus json.dumps of it, as run.py does for the output file's attrs) must not choke.
    lines = [
        "[network]",
        'hydraulics_file = "net.nc"',
        "particle_mass = 1.0",
        "[[network.sources]]",
        "reach_id = 101",
        'form = "loading"',
        "rate = 0.01",
        "start = 1979-01-01T00:00:00",
        "end = 1979-01-02T00:00:00",
    ]
    path = tmp_path / "unquoted.toml"
    path.write_text("\n".join(lines) + "\n")
    cfg = NetworkConfig.from_toml(path)
    assert isinstance(cfg.sources[0]["start"], str)
    assert isinstance(cfg.sources[0]["end"], str)
    json.dumps(cfg.to_dict())  # must not raise TypeError


def test_numpy_scalars_in_a_source_row_are_json_safe():
    # A row built programmatically (from a DataFrame, say) carries numpy scalars; to_dict() must
    # emit Python scalars so json.dumps works without a default= fallback.
    row = {
        "reach_id": np.int64(101),
        "form": "loading",
        "rate": np.float64(0.01),
        "particles": np.int32(4),
        "curve": [(np.float64(0.0), np.float64(1.0)), (np.int64(10), np.float32(2.0))],
    }
    cfg = NetworkConfig.from_dict({**MINIMAL, "sources": [row]})
    out = cfg.to_dict()["sources"][0]
    assert isinstance(out["reach_id"], int) and not isinstance(out["reach_id"], np.integer)
    assert isinstance(out["rate"], float) and isinstance(out["particles"], int)
    assert all(isinstance(t, int | float) and not isinstance(t, np.generic) for pair in out["curve"] for t in pair)
    json.dumps(cfg.to_dict())  # must not raise TypeError


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


def test_validated_sources_and_reach_subset_are_read_only():
    curve_row = {"reach_id": 102, "form": "loading", "curve": [[0.0, 0.0], [10.0, 1.0]], "particles": 2}
    cfg = NetworkConfig.from_dict({
        **MINIMAL,
        "reach_subset": {"outlet": 103},
        "sources": [*MINIMAL["sources"], curve_row],
    })
    with pytest.raises(TypeError):
        cfg.sources[0]["mass"] = 1
    with pytest.raises(TypeError):
        cfg.reach_subset["outlet"] = 1
    with pytest.raises(TypeError):  # the curve is a tuple of tuples: immutable one level down
        cfg.sources[1]["curve"][0] = (0.0, 9.0)
    d = cfg.to_dict()
    assert isinstance(d["sources"][0], dict) and isinstance(d["reach_subset"], dict)
    d["sources"][0]["mass"] = 9.0  # the emitted dicts are plain and editable
    d["sources"][1]["curve"] = [[0.0, 9.0]]  # and editing one cannot reach the frozen config
    json.dumps(d)
    assert cfg.sources[0]["mass"] == 5.0
    assert cfg.sources[1]["curve"] == ((0.0, 0.0), (10.0, 1.0))


def test_guard_rails_on_scalars():
    with pytest.raises(ValueError, match="dt must be positive"):
        NetworkConfig.from_dict({**MINIMAL, "dt": 0.0})
    with pytest.raises(ValueError, match="dt must be positive"):
        NetworkConfig.from_dict({**MINIMAL, "dt": -900.0})
    with pytest.raises(ValueError, match="particle_mass must be positive"):
        NetworkConfig.from_dict({**MINIMAL, "particle_mass": 0.0})
    with pytest.raises(ValueError, match="max_hops must be at least 1"):
        NetworkConfig.from_dict({**MINIMAL, "max_hops": 0})
    with pytest.raises(ValueError, match="particles must be a positive integer"):
        NetworkConfig.from_dict({**MINIMAL, "sources": [{**MINIMAL["sources"][0], "particles": 0}]})
    with pytest.raises(ValueError, match="dispersion scale must be positive"):
        DispersionConfig(scale=0.0)
    with pytest.raises(ValueError, match="dispersion cap must be positive"):
        DispersionConfig(cap=-1.0)


def test_particles_config_defaults_and_flattening():
    assert ParticlesConfig() == ParticlesConfig(model="passive")
    assert ParticlesConfig.from_dict({}).model == "passive"
    pc = ParticlesConfig.from_dict({"model": "tests.network.support:HalfSpeed"})
    assert pc.params == {} and pc.to_dict() == {"model": "tests.network.support:HalfSpeed"}
    with pytest.raises(ValueError, match="x"):  # the passive model takes no parameters
        ParticlesConfig.from_dict({"model": "passive", "x": 1})
    with pytest.raises(KeyError, match="passive"):
        ParticlesConfig(model="bogus")
    with pytest.raises(TypeError):
        pc.params["y"] = 2


def test_network_config_particles_table():
    cfg = NetworkConfig.from_dict(MINIMAL)
    assert cfg.particles == ParticlesConfig()
    assert cfg.to_dict()["particles"] == {"model": "passive"}
    cfg = NetworkConfig.from_dict({**MINIMAL, "particles": {"model": "tests.network.support:HalfSpeed"}})
    assert cfg.particles.model == "tests.network.support:HalfSpeed"
    assert NetworkConfig.from_dict(cfg.to_dict()) == cfg
    json.dumps(cfg.to_dict())
    with pytest.raises(ValueError, match="x"):
        NetworkConfig.from_dict({**MINIMAL, "particles": {"x": 1}})


def test_template_mentions_particles_table():
    text = get_network_config_template()
    assert "[network.particles]" in text and "drift" in text
