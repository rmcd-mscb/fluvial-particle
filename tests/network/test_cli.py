"""Tests for the network CLI and package exports."""

import pytest

import fluvial_particle
from fluvial_particle.cli import network_parser, network_serial
from tests.network.support import three_reach_dataset, write_network_file


def test_exports():
    for name in (
        "run_network_simulation",
        "NetworkConfig",
        "NetworkResults",
        "FileHydraulicsProvider",
        "Network",
        "NetworkBins",
        "estimate_particles",
        "get_network_config_template",
    ):
        assert hasattr(fluvial_particle, name) and name in fluvial_particle.__all__


def test_parser_and_init(capsys):
    p = network_parser()
    ns = p.parse_args(["s.toml", "-o", "out", "--seed", "3", "--quiet"])
    assert ns.settings_file == "s.toml" and ns.output == "out" and ns.seed == 3 and ns.quiet
    with pytest.raises(SystemExit):
        network_serial(["--init"])
    assert "[network]" in capsys.readouterr().out


def test_network_serial_runs(tmp_path):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset())
    toml = tmp_path / "run.toml"
    toml.write_text(
        "\n".join([
            "[network]",
            f'hydraulics_file = "{path}"',
            "dt = 600.0",
            "output_interval = 1200.0",
            'end_time = "1979-01-01T01:00"',
            "[network.dispersion]",
            'model = "none"',
            "[[network.sources]]",
            "reach_id = 101",
            'form = "slug"',
            "time = 0.0",
            "mass = 2.0",
            "particles = 2",
        ])
        + "\n"
    )
    network_serial([str(toml), "-o", str(tmp_path / "out"), "--seed", "1", "--quiet"])
    assert (tmp_path / "out" / "network_particles.nc").exists()
    with pytest.raises(FileNotFoundError):
        network_serial([str(tmp_path / "missing.toml"), "-o", str(tmp_path / "out")])
