"""Analytical acceptance tests for the 2D/3D solver on a uniform straight channel.

A channel with uniform velocity, depth, and shear stress has uniform diffusion coefficients, so the
random walk has closed-form answers: advection is exact, a point release becomes a Gaussian plume
with mean ``U t`` and variance ``2 K t``, and a population that starts well mixed across the channel
(or over the depth) must stay well mixed. The last two checks exercise the lateral (wet/dry) and
vertical (bed/surface) boundary rules, where bias hides.
"""

import numpy as np
import pytest
from scipy import stats

from fluvial_particle.Particles import Particles
from fluvial_particle.RiverGrid import DEFAULT_WATER_DENSITY, RiverGrid
from tests.support import write_straight_channel


pytestmark = pytest.mark.slow

FIELD_MAP_2D = {
    "bed_elevation": "bed_elevation",
    "wet_dry": "wet_dry",
    "shear_stress": "shear_stress",
    "velocity": "velocity",
    "water_surface_elevation": "water_surface_elevation",
}
FIELD_MAP_3D = {"velocity": "velocity"}

U = 1.0  # m/s
DEPTH = 2.0  # m
SHEAR = 10.0  # Pa -> u* = 0.1 m/s
USTAR = np.sqrt(SHEAR / DEFAULT_WATER_DENSITY)
BETA = 0.067
SEED = 12345


def diffusion(lev):
    """Horizontal and vertical diffusion coefficients the solver will use (McDonald & Nelson 2021)."""
    return lev + BETA * USTAR * DEPTH, BETA * USTAR * DEPTH


def run_channel(paths, x, y, z, *, dt, n_steps, lev, track3d=0, frac=None, **kwargs):
    """Drive RiverGrid + Particles directly (no file output) and return the particles."""
    river = RiverGrid(track3d, paths[0], paths[1], FIELD_MAP_2D, FIELD_MAP_3D)
    rng = np.random.RandomState(SEED)
    parts = Particles(x.size, x, y, z, rng, river, Track3D=track3d, lev=lev, beta=(BETA, BETA, BETA), **kwargs)
    parts.initial_validation(starttime=0.0, frac=frac)
    for i in range(n_steps):
        parts.move((i + 1) * dt, dt)
    return parts


def test_pure_advection_is_exact(tmp_path):
    """With K = 0 every particle moves exactly U dt per step and leaves at the boundary cell."""
    length, dx = 300.0, 5.0
    paths = write_straight_channel(tmp_path, length=length, dx=dx, shear=0.0)
    n, dt, x0 = 50, 1.0, 50.3
    x = np.full(n, x0)
    y = np.zeros(n)
    z = np.full(n, 0.5 * DEPTH)
    river = RiverGrid(0, paths[0], None, FIELD_MAP_2D, FIELD_MAP_3D)
    parts = Particles(n, x, y, z, np.random.RandomState(SEED), river, Track3D=0, lev=0.0, beta=(BETA,) * 3)
    parts.initial_validation(starttime=0.0)
    # Last cell (the downstream boundary cell) spans [length - dx, length]; entering it deactivates.
    exit_step = int(np.ceil((length - dx - x0) / (U * dt)))
    for i in range(1, exit_step + 1):
        parts.move(i * dt, dt)
        if i < exit_step:
            assert parts.in_bounds_mask is None or parts.in_bounds_mask.all(), f"early exit at step {i}"
            np.testing.assert_allclose(parts.x, x0 + i * U * dt, rtol=0.0, atol=1e-9)
            np.testing.assert_allclose(parts.y, 0.0, rtol=0.0, atol=1e-9)
    assert parts.in_bounds_mask is not None
    assert not parts.in_bounds_mask.any(), "all particles should be deactivated at the boundary cell"


def test_gaussian_plume_moments_and_normality(tmp_path):
    """A point release in a wide channel spreads as a Gaussian with mean U t and variance 2 K t."""
    n, dt, n_steps, lev = 20000, 1.0, 200, 0.25
    kh, _ = diffusion(lev)
    t_end = n_steps * dt
    paths = write_straight_channel(tmp_path, length=600.0, width=160.0, dx=5.0, dy=2.0)
    x0 = 100.0
    parts = run_channel(
        paths,
        np.full(n, x0),
        np.zeros(n),
        np.full(n, 0.5 * DEPTH),
        dt=dt,
        n_steps=n_steps,
        lev=lev,
    )
    assert parts.in_bounds_mask is None or parts.in_bounds_mask.all(), "no particle should reach the grid edge"
    var = 2.0 * kh * t_end
    for disp, mean, label in ((parts.x - x0, U * t_end, "x"), (parts.y, 0.0, "y")):
        assert abs(disp.mean() - mean) < 3.0 * np.sqrt(var / n), (
            f"{label} mean {disp.mean():.2f} vs {mean:.2f} (3 SE = {3.0 * np.sqrt(var / n):.2f})"
        )
        assert abs(disp.var() - var) < 3.0 * var * np.sqrt(2.0 / n), (
            f"{label} variance {disp.var():.4g} vs {var:.4g} (tolerance {3.0 * var * np.sqrt(2.0 / n):.4g})"
        )
        assert stats.normaltest(disp).pvalue > 0.01, f"{label} displacements are not normal"


def _uniformity_report(values, lo, hi, n_bins=20):
    """Chi-square p-value for a uniform histogram on [lo, hi] plus the edge-bin excess in sigma."""
    counts, _ = np.histogram(values, bins=n_bins, range=(lo, hi))
    expected = values.size / n_bins
    p = stats.chisquare(counts).pvalue
    edge_sigma = (counts[[0, -1]] - expected) / np.sqrt(expected)
    return p, edge_sigma, counts


def test_well_mixed_lateral_stays_uniform(tmp_path):
    """Particles released uniformly across a wet channel with dry margins remain uniform.

    The dry margin is a reflecting wall for a passive tracer; the solver implements it with
    ``handle_dry_parts`` (retry with a random-only step, then hold position). Any pile-up or
    depletion in the edge bins is bias in that rule.
    """
    n, dt, n_steps, lev = 20000, 1.0, 200, 1.0
    half = 20.0  # wet half width; sqrt(2 K t) ~ 20 m so every particle sees the walls
    paths = write_straight_channel(tmp_path, length=600.0, width=100.0, dx=5.0, dy=2.0, wet_halfwidth=half + 1e-6)
    rng = np.random.RandomState(SEED + 1)
    y0 = rng.uniform(-half, half, n)
    parts = run_channel(paths, np.full(n, 100.0), y0, np.full(n, 0.5 * DEPTH), dt=dt, n_steps=n_steps, lev=lev)
    active = parts.in_bounds_mask is None or parts.in_bounds_mask
    assert np.all(active), "no particle should leave the grid through the dry margin"
    # The cell-based wet/dry check makes cells touching a dry node dry, so the effective wall is one
    # cell inside the dry node line. Test uniformity on the interior that particles can occupy.
    y = parts.y
    lo, hi = y.min(), y.max()
    p, edge_sigma, counts = _uniformity_report(y, lo, hi)
    assert p > 1e-3, f"lateral distribution not uniform (p={p:.2e}, edge bins {edge_sigma} sigma):\n{counts}"


def test_well_mixed_vertical_stays_uniform(tmp_path):
    """Particles released uniformly over the depth remain uniform under the bed/surface rule.

    ``validate_z`` reflects particles off ``[vertbound, 1 - vertbound]`` of the depth. The earlier
    clamp left about 11% of a well-mixed column sitting exactly on the bounds after 200 s.
    """
    n, dt, n_steps, lev, vertbound = 20000, 1.0, 200, 0.25, 0.01
    _, kz = diffusion(lev)
    assert np.sqrt(2.0 * kz * n_steps * dt) > DEPTH / 2, "run must be long enough to mix over the depth"
    paths = write_straight_channel(tmp_path, length=600.0, width=100.0, dx=5.0, dy=2.0, nz=21)
    rng = np.random.RandomState(SEED + 2)
    frac0 = rng.uniform(vertbound, 1.0 - vertbound, n)
    parts = run_channel(
        paths,
        np.full(n, 100.0),
        np.zeros(n),
        np.zeros(n),
        dt=dt,
        n_steps=n_steps,
        lev=lev,
        track3d=1,
        frac=frac0,
        vertbound=vertbound,
    )
    assert parts.in_bounds_mask is None or parts.in_bounds_mask.all()
    frac = (parts.z - parts.bedelev) / parts.depth
    p, edge_sigma, counts = _uniformity_report(frac, vertbound, 1.0 - vertbound)
    assert p > 1e-3, f"vertical distribution not uniform (p={p:.2e}, edge bins {edge_sigma} sigma):\n{counts}"
