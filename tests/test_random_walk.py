"""Tests for the shared interval reflection used by both solvers."""

import numpy as np

from fluvial_particle.random_walk import reflect_interval


def test_inside_unchanged_and_single_crossing():
    x = np.array([0.5, -0.2, 1.3])
    np.testing.assert_allclose(reflect_interval(x, 0.0, 1.0), [0.5, 0.2, 0.7])


def test_multiple_crossings_fold():
    # 2.3 crosses the unit interval twice: 2.3 -> 1.7 (about hi) -> 0.3 (about lo)
    np.testing.assert_allclose(reflect_interval(np.array([2.3, -3.1]), 0.0, 1.0), [0.3, 0.9])


def test_broadcast_bounds_nan_and_degenerate():
    x = np.array([1.5, np.nan, 0.7])
    lo = np.array([1.0, 0.0, 0.5])
    hi = np.array([2.0, 1.0, 0.5])
    out = reflect_interval(x, lo, hi)
    assert out[0] == 1.5 and np.isnan(out[1]) and out[2] == 0.5


def test_returns_a_new_array_and_leaves_input_alone():
    x = np.array([-0.2, 0.5])
    out = reflect_interval(x, 0.0, 1.0)
    assert out is not x
    assert x[0] == -0.2


def test_matches_particles_validate_z_fold():
    # Same arithmetic as the fold that was inline in Particles.validate_z.
    rng = np.random.RandomState(0)
    x = rng.uniform(-3, 4, 1000)
    span = 1.0
    u = np.mod(x - 0.0, 2 * span)
    expected = np.where(u > span, 2 * span - u, u)
    np.testing.assert_allclose(reflect_interval(x, 0.0, 1.0), expected)
