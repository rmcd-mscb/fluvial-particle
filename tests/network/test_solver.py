"""Tests for NetworkSolver stepping."""

import numpy as np
import pytest

from fluvial_particle.network.config import DispersionConfig
from fluvial_particle.network.network import Network
from fluvial_particle.network.solver import (
    ACTIVE,
    EXITED,
    REMOVED,
    SETTLED,
    TERMINAL_STATUSES,
    UNRELEASED,
    NetworkSolver,
    Status,
)
from fluvial_particle.network.sources import ParticleSchedule
from tests.network.support import ArrayHydraulicsProvider, HalfSpeed, chain_dataset, three_reach_dataset


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


class FixedNormals:
    """rng stub returning preset standard normals; ``choice`` delegates to a seeded RandomState."""

    def __init__(self, values, seed=0):
        self.values = np.asarray(values, dtype=float)
        self.choices = []
        self._rs = np.random.RandomState(seed)

    def standard_normal(self, n):
        assert n == self.values.size
        return self.values.copy()

    def choice(self, a, p=None):
        out = self._rs.choice(a, p=p)
        self.choices.append((tuple(np.asarray(a).tolist()), None if p is None else tuple(np.asarray(p).tolist())))
        return out


K50 = DispersionConfig(model="constant", value=50.0)
STILL = {"velocity": [0.0, 0.0, 0.0], "flow_out": [10.0, 5.0, 80.0]}  # flow > 0 so K applies, but no advection


def test_kick_scale_is_sqrt_2k_tau():
    # sqrt(2 * 50 * 100) = 100 m per unit normal
    sol = make_solver(
        three_reach_dataset(**STILL),
        ParticleSchedule.simple(0, 500.0, 0.0),
        dt=100.0,
        dispersion=K50,
        rng=FixedNormals([0.5]),
    )
    sol.step()
    assert sol.s[0] == pytest.approx(550.0) and sol.reach[0] == 0


def test_dispersive_downstream_hop_carries_displacement():
    sol = make_solver(
        three_reach_dataset(**STILL),
        ParticleSchedule.simple(0, 900.0, 0.0),
        dt=100.0,
        dispersion=K50,
        rng=FixedNormals([2.0]),
    )
    sol.step()
    assert sol.reach[0] == 2 and sol.s[0] == pytest.approx(100.0) and sol.prev_reach[0] == 0


def test_dispersive_exit_at_end_of_step():
    sol = make_solver(
        three_reach_dataset(**STILL),
        ParticleSchedule.simple(2, 2950.0, 0.0),
        dt=100.0,
        dispersion=K50,
        rng=FixedNormals([1.0]),
    )
    sol.step()
    assert sol.status[0] == EXITED and sol.exit_time[0] == 100.0 and sol.exit_reach[0] == 2


def test_upstream_hop_returns_to_previous_reach():
    sol = make_solver(
        three_reach_dataset(**STILL),
        ParticleSchedule.simple(2, 50.0, 0.0),
        dt=100.0,
        dispersion=K50,
        rng=FixedNormals([-1.0]),
    )
    sol._status[:] = ACTIVE  # pre-place the particle with history
    sol._reach[0], sol._s[0], sol._prev_reach[0] = 2, 50.0, 0
    sol.release_time[0] = -1.0  # already released
    sol.step()
    assert sol.reach[0] == 0 and sol.s[0] == pytest.approx(950.0) and sol.prev_reach[0] == -1


def test_reflect_without_history_and_after_second_overshoot():
    sol = make_solver(
        three_reach_dataset(**STILL),
        ParticleSchedule.simple(0, 50.0, 0.0),
        dt=100.0,
        dispersion=K50,
        rng=FixedNormals([-1.0]),
    )
    sol.step()
    assert sol.reach[0] == 0 and sol.s[0] == pytest.approx(50.0)
    sol = make_solver(
        three_reach_dataset(**STILL),
        ParticleSchedule.simple(2, 50.0, 0.0),
        dt=100.0,
        dispersion=K50,
        rng=FixedNormals([-11.0]),
    )
    sol._status[:] = ACTIVE
    sol._reach[0], sol._s[0], sol._prev_reach[0] = 2, 50.0, 0
    sol.release_time[0] = -1.0
    sol.step()  # -1050: back into reach 0 at -50, then reflect to 50
    assert sol.reach[0] == 0 and sol.s[0] == pytest.approx(50.0) and sol.prev_reach[0] == -1


def test_upstream_hop_without_history_enters_a_flow_weighted_parent():
    # Reach 2 (the outlet) has parents 0 (1000 m, flow 10) and 1 (2000 m, flow 5). A particle just
    # released there has no history, so the hybrid rule sends it into one of the two parents.
    landed = {}
    for seed in range(25):
        rng = FixedNormals([-1.0], seed=seed)
        sol = make_solver(
            three_reach_dataset(**STILL),
            ParticleSchedule.simple(2, 50.0, 0.0),
            dt=100.0,
            dispersion=K50,
            rng=rng,
        )
        sol.step()  # -100 m kick from s = 50 -> s = -50
        r = int(sol.reach[0])
        assert r in {0, 1}
        assert sol.s[0] == pytest.approx(sol.network.length[r] - 50.0)
        assert sol.prev_reach[0] == -1
        assert rng.choices == [((0, 1), (10.0 / 15.0, 5.0 / 15.0))]  # weighted by this step's flow_out
        landed.setdefault(r, 0)
        landed[r] += 1
    assert set(landed) == {0, 1}, f"both parents should be reachable, saw {landed}"


def test_upstream_hop_at_a_true_headwater_reflects():
    # Reach 0 is a headwater: no history and no parents, so the particle reflects off s = 0.
    sol = make_solver(
        three_reach_dataset(**STILL),
        ParticleSchedule.simple(0, 50.0, 0.0),
        dt=100.0,
        dispersion=K50,
        rng=FixedNormals([-1.0]),
    )
    sol.step()
    assert sol.reach[0] == 0 and sol.s[0] == pytest.approx(50.0) and sol.prev_reach[0] == -1


def test_upstream_hop_uses_uniform_weights_when_parent_flows_are_zero():
    ds = three_reach_dataset(**{**STILL, "flow_out": [0.0, 0.0, 80.0]})
    rng = FixedNormals([-1.0], seed=1)
    sol = make_solver(ds, ParticleSchedule.simple(2, 50.0, 0.0), dt=100.0, dispersion=K50, rng=rng)
    sol.step()
    assert int(sol.reach[0]) in {0, 1}
    assert rng.choices == [((0, 1), None)]  # all parent flows 0 -> uniform


def _cyclic_solver(ds, schedule, *, dt, dispersion, max_hops, rng=None):
    """A solver whose topology is made cyclic after construction (Network itself rejects cycles)."""
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    net = Network(prov.static)
    net.to_index = np.array([1, 0, -1], dtype=np.int32)  # 0 <-> 1, injected past Network's validation
    return NetworkSolver(
        net,
        prov,
        schedule,
        start_time=T0,
        dt=dt,
        dispersion=dispersion,
        rng=rng or np.random.RandomState(0),
        max_hops=max_hops,
    )


def test_max_hops_raises_on_cycle():
    sol = _cyclic_solver(
        three_reach_dataset(velocity=[100.0, 100.0, 100.0]),
        ParticleSchedule.simple(0, 0.0, 0.0),
        dt=1000.0,
        dispersion=NONE,
        max_hops=3,
    )
    with pytest.raises(RuntimeError, match="max_hops"):
        sol.step()


def test_dispersive_max_hops_raises_on_cycle():
    # 10 m reaches with K = 50 and dt = 100: a kick of sqrt(2 * 50 * 100) = 100 m crosses ten reach
    # lengths, so the carry loop goes round the 0 <-> 1 cycle until max_hops stops it.
    ds = three_reach_dataset(length=[10.0, 10.0, 10.0], velocity=[0.0, 0.0, 0.0], flow_out=[10.0, 5.0, 80.0])
    sol = _cyclic_solver(
        ds,
        ParticleSchedule.simple(0, 5.0, 0.0),
        dt=100.0,
        dispersion=K50,
        max_hops=3,
        rng=FixedNormals([1.0]),
    )
    with pytest.raises(RuntimeError, match="dispersion step"):
        sol.step()


def test_cyclic_topology_is_rejected_by_network():
    static = dict(ArrayHydraulicsProvider.from_dataset(three_reach_dataset()).static)
    static["to_index"] = np.array([1, 0, -1], dtype=np.int32)
    with pytest.raises(ValueError, match=r"cycle through reach indices \[0, 1\]"):
        Network(static)


def test_seeded_reproducibility_and_conservation_with_dispersion():
    n = 200
    sch = ParticleSchedule(
        np.zeros(n, dtype=np.int32), np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n, dtype=np.int32)
    )
    a = make_solver(three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    b = make_solver(three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    for _ in range(12):
        a.step()
        b.step()
        assert np.bincount(a.status, minlength=3).sum() == n
    np.testing.assert_array_equal(a.reach, b.reach)
    np.testing.assert_allclose(a.s, b.s, equal_nan=True)
    assert np.all(
        (a.s[a.status == ACTIVE] >= 0.0) & (a.s[a.status == ACTIVE] <= a.network.length[a.reach[a.status == ACTIVE]])
    )


def test_state_arrays_are_read_only_views():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 100.0, 0.0), dt=200.0)
    for name in ("reach", "s", "prev_reach", "status", "exit_time", "exit_reach"):
        arr = getattr(sol, name)
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0
    sol.step()  # the views track the solver's own arrays
    assert sol.status[0] == ACTIVE and sol.reach[0] == 0


def test_status_enum_values_and_aliases():
    assert (Status.UNRELEASED, Status.ACTIVE, Status.EXITED) == (0, 1, 2)
    assert (UNRELEASED, ACTIVE, EXITED) == (Status.UNRELEASED, Status.ACTIVE, Status.EXITED)
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 100.0, 0.0), dt=200.0)
    assert sol.status.dtype == np.int8
    assert Status(int(sol.status[0])) is Status.UNRELEASED


def test_terminal_status_codes_and_aliases():
    assert (Status.SETTLED, Status.REMOVED) == (3, 4)
    assert (SETTLED, REMOVED) == (Status.SETTLED, Status.REMOVED)
    assert TERMINAL_STATUSES == (EXITED, SETTLED, REMOVED)
    assert all(code > ACTIVE for code in TERMINAL_STATUSES)


def test_terminate_marks_settled_and_keeps_position():
    n = 3
    sch = ParticleSchedule(
        np.zeros(n, dtype=np.int32), np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n, dtype=np.int32)
    )
    sol = make_solver(three_reach_dataset(), sch, dt=100.0)
    sol.step()
    sol.terminate(np.array([1]), SETTLED, sol.time)
    assert sol.status[1] == 3
    assert sol.exit_time[1] == sol.time
    assert sol.exit_reach[1] == sol.reach[1] == 0
    assert np.isfinite(sol.s[1]) and sol.s[1] == pytest.approx(100.0)
    sol.step()
    # a terminal particle no longer moves; the others do
    assert sol.status[1] == 3 and sol.s[1] == pytest.approx(100.0) and sol.exit_time[1] == 100.0
    assert sol.status[0] == ACTIVE and sol.s[0] == pytest.approx(200.0)


def test_terminate_rejects_non_terminal_status_and_inactive_particles():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 100.0, 0.0), dt=200.0)
    with pytest.raises(ValueError, match="terminal status"):
        sol.terminate(np.array([0]), ACTIVE, 0.0)
    with pytest.raises(ValueError, match="must be active"):
        sol.terminate(np.array([0]), SETTLED, 0.0)  # still unreleased
    sol.step()
    sol.terminate(np.array([0]), SETTLED, sol.time)
    with pytest.raises(ValueError, match="must be active"):
        sol.terminate(np.array([0]), REMOVED, sol.time)  # already terminal


def test_terminate_with_empty_index_is_a_no_op():
    sol = make_solver(three_reach_dataset(), ParticleSchedule.simple(0, 100.0, 0.0), dt=200.0)
    sol.step()
    sol.terminate(np.array([], dtype=np.int64), REMOVED, sol.time)
    assert sol.status[0] == ACTIVE and np.isnan(sol.exit_time[0])


def test_conservation_over_all_statuses():
    n = 50
    sch = ParticleSchedule(
        np.zeros(n, dtype=np.int32),
        np.linspace(0.0, 1000.0, n),
        np.linspace(0.0, 400.0, n),
        np.ones(n),
        np.zeros(n, dtype=np.int32),
    )
    sol = make_solver(three_reach_dataset(), sch, dt=300.0)
    for k in range(20):
        sol.step()
        if k == 2:
            sol.terminate(np.array([10, 11]), SETTLED, sol.time)
            sol.terminate(np.array([12]), REMOVED, sol.time)
        counts = np.bincount(sol.status, minlength=5)
        assert counts.sum() == n
    assert counts[SETTLED] == 2 and counts[REMOVED] == 1 and counts[EXITED] > 0


def make_model(cls, ds, schedule, dt, dispersion=NONE, seed=0):
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    net = Network(prov.static)
    return cls(net, prov, schedule, start_time=T0, dt=dt, dispersion=dispersion, rng=np.random.RandomState(seed))


def test_hooks_called_and_factor_halves_the_landing():
    # Two reaches of 100 m at 1 m/s, dt = 150 s, one particle released at s = 0 of reach 0.
    # Base: 150 m -> reach 1 at s = 50. Half speed: 75 m -> reach 0 at s = 75.
    ds = chain_dataset(n_reach=2, length=100.0, velocity=1.0)
    base = make_model(NetworkSolver, ds, ParticleSchedule.simple(0, 0.0, 0.0), dt=150.0)
    half = make_model(HalfSpeed, ds, ParticleSchedule.simple(0, 0.0, 0.0), dt=150.0)
    base.step()
    half.step()
    assert base.reach[0] == 1 and base.s[0] == pytest.approx(50.0)
    assert half.reach[0] == 0 and half.s[0] == pytest.approx(75.0)
    assert len(half.released) == 1 and list(half.released[0]) == [0]
    assert half.behave_calls == [(0.0, 150.0, 150.0)]
    half.step()
    assert len(half.released) == 1  # nothing new to release; the hook is not called with an empty index


def test_on_release_sees_only_the_particles_released_this_step():
    ds = chain_dataset(n_reach=2, length=100.0, velocity=1.0)
    sch = ParticleSchedule(
        np.zeros(3, dtype=np.int32),
        np.zeros(3),
        np.array([0.0, 50.0, 250.0]),
        np.ones(3),
        np.zeros(3, dtype=np.int32),
    )
    half = make_model(HalfSpeed, ds, sch, dt=100.0)
    half.step()
    half.step()
    half.step()
    assert [list(r) for r in half.released] == [[0, 1], [2]]


def test_factor_applies_across_a_hop():
    # Reach 0 at 1 m/s, reach 1 at 0.5 m/s, both 100 m; dt = 300 s at half speed: 150 m of travel in
    # reach 0 crosses at 200 s, and the 100 s left run at 0.5 * 0.5 m/s in reach 1: s = 25.
    ds = chain_dataset(n_reach=2, length=100.0, velocity=[1.0, 0.5])
    half = make_model(HalfSpeed, ds, ParticleSchedule.simple(0, 0.0, 0.0), dt=300.0)
    half.step()
    assert half.reach[0] == 1 and half.s[0] == pytest.approx(25.0) and half.prev_reach[0] == 0
    base = make_model(NetworkSolver, ds, ParticleSchedule.simple(0, 0.0, 0.0), dt=300.0)
    base.step()
    assert base.reach[0] == 1 and base.s[0] == pytest.approx(100.0)


def test_factor_applies_to_the_exit_time():
    # Reach 1 is the outlet: 100 m at 0.5 m/s * 0.5 = 400 s to exit from s = 0.
    ds = chain_dataset(n_reach=2, length=100.0, velocity=[1.0, 0.5])
    half = make_model(HalfSpeed, ds, ParticleSchedule.simple(1, 0.0, 0.0), dt=500.0)
    half.step()
    assert half.status[0] == EXITED and half.exit_time[0] == pytest.approx(400.0)


def test_behave_returning_none_equals_ones():
    class Ones(NetworkSolver):
        def behave(self, h, tau, t, dt):  # noqa: ARG002
            return np.ones(self.n)

    n = 200
    sch = ParticleSchedule(
        np.zeros(n, dtype=np.int32), np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n, dtype=np.int32)
    )
    a = make_model(NetworkSolver, three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    b = make_model(Ones, three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    for _ in range(30):
        a.step()
        b.step()
    np.testing.assert_array_equal(a.reach, b.reach)
    np.testing.assert_array_equal(a.status, b.status)
    np.testing.assert_array_equal(a.s, b.s)
    np.testing.assert_array_equal(a.exit_time, b.exit_time)


class Vertical(NetworkSolver):
    """Test model flagged as resolving the vertical (no state, no walk)."""

    resolves_vertical = True


@pytest.mark.parametrize(
    ("mode", "cls", "velocity_profile", "expected"),
    [
        ("auto", NetworkSolver, "log", False),
        ("auto", NetworkSolver, "uniform", False),
        ("auto", Vertical, "log", True),
        ("auto", Vertical, "uniform", False),
        ("off", Vertical, "log", False),
        ("off", NetworkSolver, "log", False),
        ("on", Vertical, "log", True),
        ("on", Vertical, "uniform", True),
    ],
)
def test_shear_correction_resolution(mode, cls, velocity_profile, expected):
    from fluvial_particle.network.config import VerticalDispersionConfig

    disp = DispersionConfig(shear_correction=mode, vertical=VerticalDispersionConfig(velocity_profile=velocity_profile))
    sol = make_model(cls, three_reach_dataset(), ParticleSchedule.simple(0, 0.0, 0.0), dt=100.0, dispersion=disp)
    assert sol._shear_correction_active() is expected
    none = DispersionConfig(
        model="none", shear_correction=mode, vertical=VerticalDispersionConfig(velocity_profile=velocity_profile)
    )
    sol_none = make_model(cls, three_reach_dataset(), ParticleSchedule.simple(0, 0.0, 0.0), dt=100.0, dispersion=none)
    assert sol_none._shear_correction_active() is False  # no longitudinal K to correct


def test_shear_correction_on_requires_a_vertical_model():
    disp = DispersionConfig(shear_correction="on")
    with pytest.raises(ValueError, match="shear_correction"):
        make_model(
            NetworkSolver, three_reach_dataset(), ParticleSchedule.simple(0, 0.0, 0.0), dt=100.0, dispersion=disp
        )


def test_correct_shear_is_identity_without_a_table():
    sol = make_model(Vertical, three_reach_dataset(), ParticleSchedule.simple(0, 0.0, 0.0), dt=100.0)
    assert sol._shear_table is None
    h = sol.provider.hydraulics(sol.midpoint_time())
    k = np.array([1.0, 2.0, 3.0])
    out = sol._correct_shear(k, h)
    np.testing.assert_array_equal(out, k)


def test_background_dispersion_reaches_the_kick():
    # model = "none" with a background K: the kick is sqrt(2 * 0.5 * 100) = 10 m per unit normal
    disp = DispersionConfig(model="none", background=0.5)
    sol = make_solver(
        three_reach_dataset(**STILL),
        ParticleSchedule.simple(0, 500.0, 0.0),
        dt=100.0,
        dispersion=disp,
        rng=FixedNormals([1.0]),
    )
    sol.step()
    assert sol.s[0] == pytest.approx(510.0)
