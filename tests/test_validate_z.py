"""Fast, deterministic tests for the vertical bound rule and the all-particles-exit step."""

from types import SimpleNamespace

import numpy as np
import pytest

from fluvial_particle.FallingParticles import FallingParticles
from fluvial_particle.Particles import Particles
from fluvial_particle.RiverGrid import RiverGrid
from tests.support import write_straight_channel


FIELD_MAP_2D = {
    "bed_elevation": "bed_elevation",
    "wet_dry": "wet_dry",
    "shear_stress": "shear_stress",
    "velocity": "velocity",
    "water_surface_elevation": "water_surface_elevation",
}
FIELD_MAP_3D = {"velocity": "velocity"}

VERTBOUND = 0.01
LO, HI = 0.02, 1.98  # bed 0, wse 2, vertbound 0.01


def column(n, *, bedelev=0.0, wse=2.0, vertbound=VERTBOUND):
    """Stand-in for a Particles instance with a uniform water column; validate_z needs only these."""
    bedelev = np.full(n, bedelev, dtype=float)
    wse = np.full(n, wse, dtype=float)
    return SimpleNamespace(indices=np.arange(n), bedelev=bedelev, wse=wse, depth=wse - bedelev, vertbound=vertbound)


@pytest.mark.parametrize(
    "pz, expected",
    [
        (-0.5, 0.54),  # one overshoot below the bed
        (2.5, 1.46),  # one overshoot above the surface
        (-5.0, 1.12),  # crosses the column more than once
        (100.0, 1.96),  # far above
        (LO, LO),  # exactly on the lower bound stays
        (HI, HI),  # exactly on the upper bound stays
        (1.0, 1.0),  # inside stays
    ],
    ids=["below", "above", "multi-span", "far", "on-lo", "on-hi", "inside"],
)
def test_validate_z_reflects(pz, expected):
    """The fold maps every overshoot back into [lo, hi] as a mirror reflection."""
    pz = np.array([pz])
    Particles.validate_z(column(1), pz)
    np.testing.assert_allclose(pz, [expected], atol=1e-12)


def test_validate_z_reflection_stays_in_bounds():
    """Large random overshoots always land inside [lo, hi]."""
    rng = np.random.RandomState(0)
    pz = rng.uniform(-50.0, 50.0, 1000)
    Particles.validate_z(column(pz.size), pz)
    assert np.all(pz >= LO) and np.all(pz <= HI)


def test_validate_z_leaves_nan_alone():
    """Deactivated particles carry NaN positions and must not be touched."""
    pz = np.array([np.nan, -1.0])
    Particles.validate_z(column(2), pz)
    assert np.isnan(pz[0])
    np.testing.assert_allclose(pz[1], 1.04)


def test_validate_z_pins_zero_width_column():
    """A column with no usable width (vertbound = 0.5 or zero depth) pins to the single admissible z."""
    # vertbound = 0.5 is what 2D runs force; lo == hi == mid-depth.
    pz = np.array([-5.0, 0.0, 1.0, 50.0])
    Particles.validate_z(column(4, vertbound=0.5), pz)
    np.testing.assert_allclose(pz, 1.0)
    # zero depth: lo == hi == bedelev
    pz = np.array([-1.0, 3.0])
    Particles.validate_z(column(2, bedelev=1.0, wse=1.0), pz)
    np.testing.assert_allclose(pz, 1.0)


def test_falling_particles_clamp():
    """FallingParticles clamp to the bounds instead of reflecting, so the override is load-bearing."""
    pz = np.array([-0.5, 2.5, 1.0])
    FallingParticles.validate_z(column(3), pz)
    np.testing.assert_allclose(pz, [LO, HI, 1.0])
    reflected = np.array([-0.5, 2.5, 1.0])
    Particles.validate_z(column(3), reflected)
    assert not np.allclose(pz[:2], reflected[:2])


def test_2d_run_pins_z_to_mid_depth(tmp_path):
    """A 2D run ignores the user's z and keeps particles at mid-depth from t = 0 onward."""
    paths = write_straight_channel(tmp_path, length=60.0, width=20.0, dx=5.0, dy=2.0, depth=2.0, shear=0.0)
    river = RiverGrid(0, paths[0], None, FIELD_MAP_2D, FIELD_MAP_3D)
    z = np.array([-5.0, 0.0, 1.0, 50.0])
    parts = Particles(z.size, np.full(z.size, 20.0), np.zeros(z.size), z, np.random.RandomState(0), river, Track3D=0)
    parts.initial_validation(starttime=0.0)
    np.testing.assert_allclose(parts.z, 1.0)
    np.testing.assert_allclose(parts.htabvbed, 1.0)
    parts.move(1.0, 1.0)
    np.testing.assert_allclose(parts.z, 1.0)


def test_all_particles_exit_in_same_step(tmp_path):
    """Every active particle leaving the grid in one step must not crash the wet/dry check.

    This used to raise ``ValueError: shape mismatch`` in ``Particles._is_part_wet`` because the
    2D probe output kept the previous active set's length.
    """
    paths = write_straight_channel(tmp_path, length=30.0, width=10.0, dx=5.0, dy=2.0, shear=0.0)
    river = RiverGrid(0, paths[0], None, FIELD_MAP_2D, FIELD_MAP_3D)
    n = 2
    parts = Particles(n, np.full(n, 20.0), np.zeros(n), np.ones(n), np.random.RandomState(0), river, Track3D=0)
    parts.initial_validation(starttime=0.0)
    parts.move(15.0, 15.0)
    assert parts.in_bounds_mask is not None
    assert not parts.in_bounds_mask.any()
