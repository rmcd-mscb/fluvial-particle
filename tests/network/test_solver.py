"""Tests for NetworkSolver stepping."""

import numpy as np
import pytest

from fluvial_particle.network.config import DispersionConfig
from fluvial_particle.network.network import Network
from fluvial_particle.network.solver import ACTIVE, EXITED, UNRELEASED, NetworkSolver
from fluvial_particle.network.sources import ParticleSchedule
from tests.network.support import ArrayHydraulicsProvider, chain_dataset, three_reach_dataset


T0 = np.datetime64("1979-01-01", "ns")
NONE = DispersionConfig(model="none")


def make_solver(ds, schedule, dt, dispersion=NONE, seed=0, max_hops=1000, rng=None):
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    net = Network(prov.static)
    return NetworkSolver(
        net,
        prov,
        schedule,
        start_time=T0,
        dt=dt,
        dispersion=dispersion,
        rng=rng or np.random.RandomState(seed),
        max_hops=max_hops,
    )


def test_release_at_start_and_plain_advection():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 100.0, 0.0), dt=200.0)
    assert sol.status[0] == UNRELEASED and sol.reach[0] == -1 and np.isnan(sol.s[0])
    sol.step()
    assert sol.status[0] == ACTIVE and sol.reach[0] == 0
    assert sol.s[0] == pytest.approx(300.0)  # v = 1 m/s
    assert sol.time == 200.0
    assert sol.prev_reach[0] == -1


def test_one_hop_with_time_carry():
    # reach 0 (1000 m, 1 m/s) -> reach 2 (2 m/s): 900 + 200 overshoots by 100 m = 100 s left -> 200 m into reach 2
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 900.0, 0.0), dt=200.0)
    sol.step()
    assert sol.reach[0] == 2 and sol.s[0] == pytest.approx(200.0) and sol.prev_reach[0] == 0


def test_two_hops_in_one_step():
    ds = chain_dataset(n_reach=4, length=1000.0, velocity=[1.0, 2.0, 4.0, 1.0])
    sol = make_solver(ds, ParticleSchedule.simple(0, 0.0, 0.0), dt=1600.0)  # 1000 s + 500 s + 100 s at 4 m/s
    sol.step()
    assert sol.reach[0] == 2 and sol.s[0] == pytest.approx(400.0) and sol.prev_reach[0] == 1


def test_hop_into_zero_velocity_reach_waits():
    ds = three_reach_dataset(velocity=[1.0, 0.5, 0.0], flow_out=[10.0, 5.0, 0.0])
    sol = make_solver(ds, ParticleSchedule.simple(0, 900.0, 0.0), dt=200.0)
    sol.step()
    assert sol.reach[0] == 2 and sol.s[0] == 0.0 and sol.status[0] == ACTIVE
    sol.step()
    assert sol.s[0] == 0.0


def test_exit_records_exact_time():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(2, 2900.0, 0.0), dt=100.0)
    sol.step()  # 2 m/s: reaches the end after 50 s
    assert sol.status[0] == EXITED and sol.reach[0] == -1 and np.isnan(sol.s[0])
    assert sol.exit_time[0] == pytest.approx(50.0) and sol.exit_reach[0] == 2
    sol.step()  # exited particles stay put
    assert sol.exit_time[0] == pytest.approx(50.0)


def test_mid_step_release_uses_partial_time():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 0.0, 50.0), dt=100.0)
    sol.step()
    assert sol.status[0] == ACTIVE and sol.s[0] == pytest.approx(50.0)
    later = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 0.0, 250.0), dt=100.0)
    later.step()
    later.step()
    assert later.status[0] == UNRELEASED
    later.step()
    assert later.status[0] == ACTIVE and later.s[0] == pytest.approx(50.0)


def test_mass_conservation_and_vectorized_batch():
    n = 50
    sch = ParticleSchedule(
        np.zeros(n, dtype=np.int32),
        np.linspace(0.0, 1000.0, n),
        np.linspace(0.0, 400.0, n),
        np.ones(n),
        np.zeros(n, dtype=np.int32),
    )
    sol = make_solver(three_reach_dataset(), sch, dt=300.0)
    for _ in range(8):  # 2400 s: the earliest particles have exited, the latest are still in reach 2
        sol.step()
        counts = np.bincount(sol.status, minlength=3)
        assert counts.sum() == n
    assert (sol.status == EXITED).sum() > 0 and (sol.status == ACTIVE).sum() > 0


def test_distance_along_helper():
    ds = chain_dataset(n_reach=3, length=1000.0, velocity=1.0)
    sol = make_solver(ds, ParticleSchedule.simple(0, 0.0, 0.0), dt=1500.0)
    sol.step()
    cum_before = np.array([0.0, 1000.0, 2000.0])
    assert sol.distance_along(cum_before)[0] == pytest.approx(1500.0)
