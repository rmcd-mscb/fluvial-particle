"""Tests for declared particle state, the model registry, and the drift model."""

import numpy as np
import pytest

from fluvial_particle.network.config import DispersionConfig
from fluvial_particle.network.network import Network
from fluvial_particle.network.particles import PARTICLE_MODELS, StateVar, resolve_model
from fluvial_particle.network.solver import NetworkSolver
from fluvial_particle.network.sources import ParticleSchedule
from tests.network.support import ArrayHydraulicsProvider, HalfSpeed, three_reach_dataset


T0 = np.datetime64("1979-01-01", "ns")
NONE = DispersionConfig(model="none")


class TwoState(NetworkSolver):
    """Test model declaring a scalar and a labelled (2,) state."""

    STATE = (
        StateVar("zeta", units="1", long_name="relative elevation"),
        StateVar("c", shape=(2,), dim="constituent", labels=("a", "b"), kind="extensive", units="kg"),
    )


def make(cls, ds, schedule, dt=100.0, dispersion=NONE, seed=0, **kwargs):
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    net = Network(prov.static)
    return cls(
        net, prov, schedule, start_time=T0, dt=dt, dispersion=dispersion, rng=np.random.RandomState(seed), **kwargs
    )


def slug(n, reach=0):
    return ParticleSchedule(
        np.full(n, reach, dtype=np.int32), np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n, dtype=np.int32)
    )


# ---- StateVar and allocation --------------------------------------------------
def test_statevar_defaults_and_frozen():
    sv = StateVar("zeta")
    assert (sv.dtype, sv.shape, sv.output, sv.units, sv.long_name, sv.kind, sv.dim, sv.labels) == (
        "f8",
        (),
        True,
        "",
        "",
        None,
        None,
        None,
    )
    assert np.isnan(sv.fill)
    with pytest.raises(AttributeError):
        sv.name = "x"  # type: ignore[misc]


def test_statevar_validates_kind_and_vector_dim():
    with pytest.raises(ValueError, match="kind"):
        StateVar("x", kind="weird")
    with pytest.raises(ValueError, match="dim"):
        StateVar("x", shape=(2,))
    with pytest.raises(ValueError, match="labels"):
        StateVar("x", shape=(2,), dim="k", labels=("only_one",))
    with pytest.raises(ValueError, match="shape"):
        StateVar("x", shape=(2, 3), dim="k")


def test_base_solver_declares_no_state():
    sol = make(NetworkSolver, three_reach_dataset(), slug(3))
    assert sol.state_specs == () and dict(sol.state) == {}


def test_state_is_allocated_per_particle_and_read_only():
    sol = make(TwoState, three_reach_dataset(), slug(4))
    assert sol.state_specs == TwoState.STATE
    assert set(sol.state) == {"zeta", "c"}
    assert sol.state["zeta"].shape == (4,) and sol.state["c"].shape == (4, 2)
    assert np.isnan(sol.state["zeta"]).all() and np.isnan(sol.state["c"]).all()
    assert not sol.state["zeta"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        sol.state["zeta"][0] = 1.0
    sol._state["zeta"][0] = 0.25  # the model's write path; the view tracks it
    assert sol.state["zeta"][0] == 0.25


def test_state_dtype_and_fill_are_honoured():
    class Flags(NetworkSolver):
        STATE = (StateVar("stage", dtype="i2", fill=-1),)

    sol = make(Flags, three_reach_dataset(), slug(2))
    assert sol.state["stage"].dtype == np.int16 and list(sol.state["stage"]) == [-1, -1]


@pytest.mark.parametrize("name", ["reach", "s", "status", "mass", "exit_time", "reach_index", "time"])
def test_reserved_state_name_raises(name):
    class Bad(NetworkSolver):
        STATE = (StateVar(name),)

    with pytest.raises(ValueError, match="reserved"):
        make(Bad, three_reach_dataset(), slug(1))


def test_duplicate_state_name_raises():
    class Dup(NetworkSolver):
        STATE = (StateVar("x"), StateVar("x"))

    with pytest.raises(ValueError, match="duplicate"):
        make(Dup, three_reach_dataset(), slug(1))


# ---- registry and params -------------------------------------------------------
def test_registry_names_and_lookup():
    assert PARTICLE_MODELS["passive"] is NetworkSolver
    assert resolve_model("passive") is NetworkSolver
    assert resolve_model("tests.network.support:HalfSpeed") is HalfSpeed


def test_registry_unknown_name_lists_known_models():
    with pytest.raises(KeyError, match="passive"):
        resolve_model("bogus")


def test_registry_rejects_non_subclass_and_bad_paths():
    with pytest.raises(TypeError, match="NetworkSolver"):
        resolve_model("tests.network.support:NotASolver")
    with pytest.raises(ImportError):
        resolve_model("tests.network.support:Missing")
    with pytest.raises(ImportError):
        resolve_model("no.such.module:Thing")


def test_base_validate_params_rejects_every_key():
    assert NetworkSolver.validate_params({}) == {}
    with pytest.raises(ValueError, match="x"):
        NetworkSolver.validate_params({"x": 1})
    with pytest.raises(ValueError, match="x"):
        make(NetworkSolver, three_reach_dataset(), slug(1), params={"x": 1})
    sol = make(NetworkSolver, three_reach_dataset(), slug(1))
    assert sol.params == {} and NetworkSolver.resolves_vertical is False and sol.resolves_vertical is False


# ---- DriftParticles -------------------------------------------------------------
from fluvial_particle.network.config import VerticalDispersionConfig  # noqa: E402
from fluvial_particle.network.particles import DriftParticles  # noqa: E402
from fluvial_particle.network.solver import ACTIVE, SETTLED  # noqa: E402
from fluvial_particle.network.vertical import VerticalProfiles  # noqa: E402
from tests.network.support import chain_dataset  # noqa: E402


NO_MIXING = DispersionConfig(
    model="none", vertical=VerticalDispersionConfig(profile="value", value=0.0, velocity_profile="uniform")
)
NO_MIXING_LOG = DispersionConfig(model="none", vertical=VerticalDispersionConfig(profile="value", value=0.0))


def test_drift_is_registered_and_resolves_the_vertical():
    assert PARTICLE_MODELS["drift"] is DriftParticles and resolve_model("drift") is DriftParticles
    assert DriftParticles.resolves_vertical is True
    assert [s.name for s in DriftParticles.STATE] == ["zeta", "velocity_factor"]
    assert DriftParticles.STATE[1].output is False


def test_drift_validate_params_defaults_and_errors():
    p = DriftParticles.validate_params({})
    assert p == {
        "settling_velocity": 0.0,
        "swim_velocity": 0.0,
        "diel": None,
        "deposition_velocity": 0.0,
        "critical_ustar": None,
        "zeta_min": 0.001,
        "max_substeps": 500,
        "initial_zeta": "uniform",
    }
    assert DriftParticles.validate_params({"initial_zeta": 0.3})["initial_zeta"] == 0.3
    diel = DriftParticles.validate_params({"diel": {"amplitude": 0.01, "period": 86400.0}})["diel"]
    assert diel == {"amplitude": 0.01, "period": 86400.0, "phase": 0.0}
    with pytest.raises(ValueError, match="unknown"):
        DriftParticles.validate_params({"bogus": 1})
    with pytest.raises(ValueError, match="deposition_velocity"):
        DriftParticles.validate_params({"deposition_velocity": -1.0})
    with pytest.raises(ValueError, match="settling_velocity"):
        DriftParticles.validate_params({"settling_velocity": -1.0})
    with pytest.raises(ValueError, match="critical_ustar"):
        DriftParticles.validate_params({"critical_ustar": 0.0})
    with pytest.raises(ValueError, match="zeta_min"):
        DriftParticles.validate_params({"zeta_min": 0.5})
    with pytest.raises(ValueError, match="max_substeps"):
        DriftParticles.validate_params({"max_substeps": 0})
    with pytest.raises(ValueError, match="initial_zeta"):
        DriftParticles.validate_params({"initial_zeta": 0.9995})
    with pytest.raises(ValueError, match="initial_zeta"):
        DriftParticles.validate_params({"initial_zeta": "surface"})
    with pytest.raises(ValueError, match="diel"):
        DriftParticles.validate_params({"diel": {"amplitude": 0.01}})
    with pytest.raises(ValueError, match="diel"):
        DriftParticles.validate_params({"diel": {"amplitude": 0.01, "period": 1.0, "tilt": 0.0}})
    with pytest.raises(ValueError, match="period"):
        DriftParticles.validate_params({"diel": {"amplitude": 0.01, "period": 0.0}})


def test_drift_on_release_initializes_zeta():
    ds = chain_dataset(n_reach=2, length=1000.0, velocity=1.0)
    sol = make(DriftParticles, ds, slug(200), dt=10.0, dispersion=NO_MIXING, params={"zeta_min": 0.01})
    assert np.isnan(sol.state["zeta"]).all()
    sol.step()
    z = sol.state["zeta"]
    assert (z >= 0.01).all() and (z <= 0.99).all() and z.std() > 0.1
    fixed = make(DriftParticles, ds, slug(3), dt=10.0, dispersion=NO_MIXING, params={"initial_zeta": 0.3})
    fixed.step()
    np.testing.assert_allclose(fixed.state["zeta"], 0.3)


def test_drift_zeta_is_preserved_across_a_hop_without_mixing():
    ds = chain_dataset(n_reach=2, length=100.0, velocity=1.0)
    sol = make(DriftParticles, ds, slug(1), dt=150.0, dispersion=NO_MIXING, params={"initial_zeta": 0.3})
    sol.step()
    assert sol.reach[0] == 1 and sol.s[0] == pytest.approx(50.0)
    assert sol.state["zeta"][0] == 0.3
    assert sol.last_substeps == 1


def test_drift_velocity_factor_moves_the_particle_at_its_depth():
    ds = chain_dataset(n_reach=2, length=10000.0, velocity=1.0)
    sol = make(DriftParticles, ds, slug(1), dt=100.0, dispersion=NO_MIXING_LOG, params={"initial_zeta": 0.5})
    sol.step()
    h = sol.provider.hydraulics(sol.start_time)
    vp = VerticalProfiles(NO_MIXING_LOG.vertical, 0.001)
    f = vp.velocity_factor(np.array([0.5]), h["ustar"][:1], h["velocity"][:1])[0]
    assert f != pytest.approx(1.0)
    assert sol.s[0] == pytest.approx(f * 1.0 * 100.0)
    assert sol.state["velocity_factor"][0] == pytest.approx(f)


def test_drift_settles_with_an_absorbing_bed_and_not_above_critical_shear():
    ds = three_reach_dataset()  # depth 1 m, ustar ~ 0.099 m/s
    params = {"deposition_velocity": 1e9, "settling_velocity": 0.05}
    sol = make(DriftParticles, ds, slug(20), dt=100.0, params=params)
    sol.step()  # settling 5 m in a 1 m column: every particle touches the bed and sticks
    assert (sol.status == SETTLED).all()
    assert np.isfinite(sol.s).all() and (sol.exit_reach == sol.reach).all() and (sol.exit_time == 100.0).all()
    np.testing.assert_allclose(sol.state["zeta"], 0.001)
    s_before = sol.s.copy()
    sol.step()
    np.testing.assert_array_equal(sol.s, s_before)  # a settled particle no longer moves
    assert (sol.status == SETTLED).all()
    held = make(DriftParticles, ds, slug(20), dt=100.0, params={**params, "critical_ustar": 0.05})
    held.step()
    assert (held.status == ACTIVE).all()  # Krone: no deposition above the critical shear
    assert (held.state["zeta"] >= 0.001).all()


def test_drift_deposition_is_partial_for_a_finite_deposition_velocity():
    ds = three_reach_dataset()
    # k_d = 1e-5 m/s at Kz(zeta_min) ~ 4e-5 m2/s and 10 s sub-steps is a per-contact probability of
    # about 1 percent; every particle reaches the bed within the step but few stick.
    sol = make(DriftParticles, ds, slug(400), dt=100.0, params={"deposition_velocity": 1e-5, "settling_velocity": 0.05})
    sol.step()
    settled = int((sol.status == SETTLED).sum())
    assert 0 < settled < 200
    assert ((sol.status == ACTIVE) | (sol.status == SETTLED)).all()


def test_drift_shear_table_follows_the_dispersion_config():
    ds = three_reach_dataset()
    sol = make(DriftParticles, ds, slug(1), dt=100.0)
    assert sol._shear_table is sol.profiles
    uniform = DispersionConfig(vertical=VerticalDispersionConfig(velocity_profile="uniform"))
    assert make(DriftParticles, ds, slug(1), dt=100.0, dispersion=uniform)._shear_table is None
    off = DispersionConfig(shear_correction="off")
    assert make(DriftParticles, ds, slug(1), dt=100.0, dispersion=off)._shear_table is None
    on = DispersionConfig(shear_correction="on", vertical=VerticalDispersionConfig(velocity_profile="uniform"))
    assert make(DriftParticles, ds, slug(1), dt=100.0, dispersion=on)._shear_table is not None
    with pytest.raises(ValueError, match="shear_correction"):
        make(NetworkSolver, ds, slug(1), dt=100.0, dispersion=DispersionConfig(shear_correction="on"))


def test_drift_shear_correction_reduces_the_kick():
    # With the correction the Fischer K on reach 0 loses c * ustar * h; K = 10 by construction.
    ds = chain_dataset(n_reach=2, length=1e6, velocity=1.0, k_target=10.0)
    h = ArrayHydraulicsProvider.from_dataset(ds).hydraulics(T0)
    sol = make(DriftParticles, ds, slug(1), dt=100.0, dispersion=DispersionConfig())
    k = sol._correct_shear(np.array([10.0, 10.0]), h)
    c = sol.profiles.shear_coefficient(h["ustar"], h["velocity"], h["depth"])
    np.testing.assert_allclose(k, 10.0 - c * h["ustar"] * h["depth"])
    assert 0.0 < k[0] < 10.0


def test_drift_diel_velocity_is_sinusoidal():
    sol = make(
        DriftParticles,
        three_reach_dataset(),
        slug(1),
        dt=100.0,
        params={"settling_velocity": 0.01, "swim_velocity": 0.004, "diel": {"amplitude": 0.002, "period": 100.0}},
    )
    assert sol._vertical_velocity(0.0) == pytest.approx(0.006)
    assert sol._vertical_velocity(25.0) == pytest.approx(0.008)
    assert sol._vertical_velocity(75.0) == pytest.approx(0.004)


def test_drift_dry_reach_particles_do_not_walk():
    ds = three_reach_dataset(velocity=[1.0, 0.0, 2.0], depth=[1.0, 0.0, 2.0], flow_out=[10.0, 0.0, 80.0])
    sol = make(DriftParticles, ds, slug(5, reach=1), dt=100.0, params={"initial_zeta": 0.4})
    sol.step()
    assert (sol.status == ACTIVE).all()
    np.testing.assert_allclose(sol.state["zeta"], 0.4)
    assert (sol.s == 0.0).all()


def test_drift_does_not_change_the_passive_model():
    n = 100
    sch = slug(n)
    a = make(NetworkSolver, three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    drift = make(DriftParticles, three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    for _ in range(10):
        a.step()
        drift.step()
    b = make(PARTICLE_MODELS["passive"], three_reach_dataset(), sch, dt=300.0, dispersion=DispersionConfig(), seed=7)
    for _ in range(10):
        b.step()
    np.testing.assert_array_equal(a.s, b.s)
    np.testing.assert_array_equal(a.status, b.status)
    assert not np.array_equal(a.s, drift.s)


def test_drift_diagnostics_report_substeps_and_shear():
    sol = make(DriftParticles, three_reach_dataset(), slug(1), dt=900.0)
    lines = sol.diagnostics(sol.provider.hydraulics(sol.start_time))
    text = "\n".join(lines)
    assert "sub-steps" in text and "shear correction" in text and "active" in text
    assert make(NetworkSolver, three_reach_dataset(), slug(1), dt=900.0).diagnostics({}) == []
