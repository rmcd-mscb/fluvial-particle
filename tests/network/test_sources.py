"""Tests for source expansion into a particle schedule."""

import numpy as np
import pytest

from fluvial_particle.network.config import NetworkConfig
from fluvial_particle.network.network import Network
from fluvial_particle.network.sources import ParticleSchedule, estimate_particles, expand_sources, seconds_from_start
from tests.network.support import ArrayHydraulicsProvider, three_reach_dataset


T0 = np.datetime64("1979-01-01", "ns")
T1 = np.datetime64("1979-01-03", "ns")
DAY = 86400.0


@pytest.fixture
def env():
    ds = three_reach_dataset(flow_in=[10.0, 5.0, 60.0], flow_out=[10.0, 5.0, 80.0])
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    return Network(prov.static), prov


def expand(env, sources, particle_mass=None, seed=0):
    net, prov = env
    return expand_sources(
        sources, net, prov, start_time=T0, end_time=T1, particle_mass=particle_mass, rng=np.random.RandomState(seed)
    )


def test_seconds_from_start():
    assert seconds_from_start(30.0, T0) == 30.0
    assert seconds_from_start("1979-01-02", T0) == DAY


def test_slug_global_particle_mass(env):
    sch = expand(env, [{"reach_id": 101, "form": "slug", "time": 10.0, "mass": 2.5}], particle_mass=1.0)
    assert sch.n == 2  # round(2.5) = 2, last absorbs remainder
    np.testing.assert_allclose(sch.mass, [1.0, 1.5])
    assert list(sch.release_reach) == [0, 0]
    assert list(sch.release_time) == [10.0, 10.0]
    assert list(sch.release_s) == [0.0, 0.0]
    assert list(sch.source_index) == [0, 0]


def test_slug_fixed_particles_and_position(env):
    sch = expand(
        env, [{"reach_id": 102, "form": "slug", "time": "1979-01-02", "mass": 3.0, "particles": 4, "s_frac": 0.25}]
    )
    assert sch.n == 4
    np.testing.assert_allclose(sch.mass, 0.75)
    np.testing.assert_allclose(sch.release_s, 500.0)
    assert list(sch.release_time) == [DAY] * 4
    sch = expand(env, [{"reach_id": 102, "form": "slug", "time": 0.0, "mass": 3.0, "particles": 1, "s": 1999.0}])
    assert sch.release_s[0] == 1999.0


def test_constant_loading_even_spacing(env):
    row = {"reach_id": 101, "form": "loading", "rate": 0.01, "start": 0.0, "end": 1000.0, "particles": 4}
    sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 10.0)
    np.testing.assert_allclose(sch.release_time, [125.0, 375.0, 625.0, 875.0])


def test_ramp_curve_quantiles(env):
    # rate rises linearly 0 -> 1 over 1000 s: cumulative M(t) = t^2 / 2000, total 500
    row = {"reach_id": 101, "form": "loading", "curve": [[0.0, 0.0], [1000.0, 1.0]], "particles": 2}
    sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 500.0)
    # quantiles at M = 125 and 375 -> t = sqrt(2000 * M)
    np.testing.assert_allclose(sch.release_time, np.sqrt(2000.0 * np.array([125.0, 375.0])))


def test_concentration_uses_interpolated_flow(env):
    # reach 103: flow_in 60, flow_out 80, s_frac 0.5 -> Q = 70; C = 2 for 100 s -> M = 14000
    row = {
        "reach_id": 103,
        "form": "concentration",
        "value": 2.0,
        "start": 0.0,
        "end": 100.0,
        "s_frac": 0.5,
        "particles": 7,
    }
    sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 14000.0)
    row = {"reach_id": 103, "form": "concentration", "curve": [[0.0, 2.0], [100.0, 2.0]], "particles": 7}
    sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 100.0 * 2.0 * 60.0)  # s = 0 -> flow_in


def test_poisson_spacing_reproducible(env):
    row = {"reach_id": 101, "form": "loading", "rate": 0.01, "start": 0.0, "end": 1000.0, "spacing": "poisson"}
    a = expand(env, [row], particle_mass=0.5, seed=3)
    b = expand(env, [row], particle_mass=0.5, seed=3)
    assert a.n == b.n and a.n > 0
    np.testing.assert_allclose(a.release_time, b.release_time)
    assert np.all(np.diff(a.release_time) >= 0)
    np.testing.assert_allclose(a.mass.sum(), 10.0)


def test_multiple_sources_concatenate(env):
    sch = expand(
        env,
        [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 2},
            {"reach_id": 102, "form": "slug", "time": 5.0, "mass": 1.0, "particles": 3},
        ],
    )
    assert sch.n == 5
    assert list(sch.source_index) == [0, 0, 1, 1, 1]
    part = sch.slice(1, 4)
    assert part.n == 3 and list(part.source_index) == [0, 1, 1]


def test_errors_and_truncation(env):
    with pytest.raises(KeyError):
        expand(env, [{"reach_id": 999, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 1}])
    with pytest.raises(ValueError, match="s"):
        expand(env, [{"reach_id": 101, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 1, "s": 5000.0}])
    with pytest.raises(ValueError, match="window"):
        expand(env, [{"reach_id": 101, "form": "slug", "time": 3 * DAY, "mass": 1.0, "particles": 1}])
    with pytest.raises(ValueError, match="mass"):
        expand(env, [{"reach_id": 101, "form": "loading", "rate": 0.0, "start": 0.0, "end": 10.0, "particles": 1}])
    with pytest.warns(UserWarning, match="window"):
        sch = expand(
            env, [{"reach_id": 101, "form": "loading", "rate": 1.0, "start": DAY, "end": 3 * DAY, "particles": 2}]
        )
    np.testing.assert_allclose(sch.mass.sum(), DAY)  # only the day inside the window


def test_missing_particles_and_particle_mass_raises(env):
    with pytest.raises(ValueError, match=r"sources\[0\] needs particles, or a global particle_mass"):
        expand(env, [{"reach_id": 101, "form": "slug", "time": 0.0, "mass": 1.0}], particle_mass=None)


def test_schedule_simple_and_concat():
    a = ParticleSchedule.simple(reach=0, s=10.0, time=0.0)
    b = ParticleSchedule.simple(reach=1, s=0.0, time=5.0, mass=2.0, source_index=1)
    c = ParticleSchedule.concat([a, b])
    assert c.n == 2 and list(c.mass) == [1.0, 2.0]


def test_tabulated_curve_truncation_warns_and_conserves_mass(env):
    # A curve whose points straddle the end of the run window: the part inside carries mass, the
    # part outside is dropped with a warning rather than silently.
    row = {"reach_id": 101, "form": "loading", "curve": [[0.0, 1.0], [2 * DAY, 1.0], [3 * DAY, 1.0]], "particles": 4}
    with pytest.warns(UserWarning, match="truncated"):
        sch = expand(env, [row])
    np.testing.assert_allclose(sch.mass.sum(), 2 * DAY, rtol=1e-12)  # only the two days inside


def test_curve_entirely_outside_the_window_raises(env):
    row = {
        "reach_id": 101,
        "form": "loading",
        "curve": [[3 * DAY, 1.0], [4 * DAY, 1.0]],
        "particles": 4,
    }
    with pytest.warns(UserWarning, match="truncated"), pytest.raises(ValueError, match="zero mass"):
        expand(env, [row])


def test_mass_conservation_is_exact(env):
    sch = expand(env, [{"reach_id": 101, "form": "slug", "time": 0.0, "mass": 7.0}], particle_mass=0.3)
    np.testing.assert_allclose(sch.mass.sum(), 7.0, rtol=1e-12)
    sch = expand(env, [{"reach_id": 101, "form": "loading", "rate": 0.01, "end": 1000.0, "particles": 7}])
    np.testing.assert_allclose(sch.mass.sum(), 10.0, rtol=1e-12)


def test_estimate_particles_surfaces_truncation_warning(env):
    _, prov = env
    cfg = NetworkConfig.from_dict({
        "hydraulics_file": "unused.nc",
        "sources": [{"reach_id": 101, "form": "loading", "rate": 0.01, "start": 0.0, "end": 3 * DAY, "particles": 1}],
    })
    with pytest.warns(UserWarning, match="truncated"):
        estimate_particles(cfg, prov)


def test_estimate_particles_zero_velocity_raises():
    ds = three_reach_dataset(velocity=[0.0, 0.5, 2.0], flow_out=[10.0, 5.0, 80.0])
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    cfg = NetworkConfig.from_dict({
        "hydraulics_file": "unused.nc",
        "sources": [{"reach_id": 101, "form": "loading", "rate": 0.01, "start": 0.0, "end": DAY, "particles": 1}],
    })
    with pytest.raises(ValueError, match=r"sources\[0\] on reach 101 has mean velocity"):
        estimate_particles(cfg, prov)


def test_estimate_particles_hand_values(env):
    _, prov = env
    cfg = NetworkConfig.from_dict({
        "hydraulics_file": "unused.nc",
        "dispersion": {"model": "constant", "value": 10.0},
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 100.0, "particles": 1},
            {"reach_id": 101, "form": "loading", "rate": 0.01, "start": 0.0, "end": DAY, "particles": 1},
        ],
    })
    df = estimate_particles(cfg, prov, target_per_bin=100.0, bin_length=100.0, reference_travel_time=DAY)
    sigma = np.sqrt(2 * 10.0 * DAY)
    n_slug = int(np.ceil(100.0 * sigma * np.sqrt(2 * np.pi) / 100.0))
    assert df.loc[0, "particles"] == n_slug
    assert df.loc[0, "particle_mass"] == pytest.approx(100.0 / n_slug)
    # continuous: m = rate * bin / (v * target) = 0.01 * 100 / (1.0 * 100) = 0.01; N = M / m = 864 / 0.01
    assert df.loc[1, "particle_mass"] == pytest.approx(0.01)
    assert df.loc[1, "particles"] == 86400
    assert list(df.columns) == ["source", "form", "reach_id", "total_mass", "particle_mass", "particles"]
