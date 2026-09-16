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
