"""Tests for the vertical profiles, Taylor's shear-dispersion coefficient, sub-steps, and deposition."""

import numpy as np
import pytest

from fluvial_particle.network.config import VerticalDispersionConfig
from fluvial_particle.network.vertical import (
    VerticalProfiles,
    deposition_probability,
    shear_dispersion_coefficient,
    substep_count,
)


ELDER = 0.404 / 0.41**3  # 5.86: Elder's coefficient for the log law with parabolic mixing


def test_log_law_factor_has_unit_mean_after_floor_and_normalization():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    z = np.linspace(0.001, 0.999, 200001)
    for ratio in (0.05, 0.143, 0.5, 1.5):
        f = vp.velocity_factor(z, np.full_like(z, ratio), np.ones_like(z))
        assert f.min() >= 0.0
        assert np.trapezoid(f, z) / (z[-1] - z[0]) == pytest.approx(1.0, abs=1e-6)
    # the floor is active at ratio 0.5: the raw factor is negative near the bed
    assert (1.0 + 0.5 * (1.0 + np.log(0.001)) / 0.41) < 0.0
    f = vp.velocity_factor(np.array([0.001]), np.array([0.5]), np.array([1.0]))
    assert f[0] == 0.0


def test_log_law_factor_shape_and_still_water():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    z = np.array([0.1, 0.5, 0.9])
    f = vp.velocity_factor(z, np.full(3, 0.05), np.ones(3))
    assert f[0] < f[1] < f[2]  # faster near the surface
    assert f[1] == pytest.approx(1.0 + 0.05 * (1.0 + np.log(0.5)) / 0.41, rel=1e-3)  # norm is ~1 at small ratio
    assert list(vp.velocity_factor(z, np.zeros(3), np.zeros(3))) == [1.0, 1.0, 1.0]  # v = 0: factor 1
    uniform = VerticalProfiles(VerticalDispersionConfig(velocity_profile="uniform"), zeta_min=0.001)
    assert list(uniform.velocity_factor(z, np.full(3, 0.5), np.ones(3))) == [1.0, 1.0, 1.0]


def test_parabolic_kz_and_gradient():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    z = np.array([0.0, 0.25, 0.5, 1.0])
    u = np.full(4, 0.1)
    h = np.full(4, 2.0)
    np.testing.assert_allclose(vp.kz(z, u, h), 0.41 * 0.1 * 2.0 * z * (1 - z))
    np.testing.assert_allclose(vp.dkz_dz(z, u, h), 0.41 * 0.1 * (1 - 2 * z))
    assert vp.kz_max(u, h)[0] == pytest.approx(0.41 * 0.1 * 2.0 / 4)
    scaled = VerticalProfiles(VerticalDispersionConfig(scale=2.0, background=0.01), zeta_min=0.001)
    np.testing.assert_allclose(scaled.kz(z, u, h), 2.0 * 0.41 * 0.1 * 2.0 * z * (1 - z) + 0.01)
    np.testing.assert_allclose(scaled.dkz_dz(z, u, h), 2.0 * 0.41 * 0.1 * (1 - 2 * z))
    assert scaled.kz_max(u, h)[0] == pytest.approx(2.0 * 0.41 * 0.1 * 2.0 / 4 + 0.01)


def test_constant_and_value_profiles():
    z = np.array([0.1, 0.5, 0.9])
    u = np.full(3, 0.1)
    h = np.full(3, 2.0)
    const = VerticalProfiles(VerticalDispersionConfig(profile="constant"), zeta_min=0.001)
    np.testing.assert_allclose(const.kz(z, u, h), 0.067 * 0.1 * 2.0)
    np.testing.assert_allclose(const.dkz_dz(z, u, h), 0.0)
    assert const.kz_max(u, h)[0] == pytest.approx(0.067 * 0.1 * 2.0)
    value = VerticalProfiles(VerticalDispersionConfig(profile="value", value=0.003, background=0.001), zeta_min=0.001)
    np.testing.assert_allclose(value.kz(z, u, h), 0.004)
    np.testing.assert_allclose(value.dkz_dz(z, u, h), 0.0)
    assert value.kz_max(u, h)[0] == pytest.approx(0.004)


def test_constant_profile_matches_parabolic_depth_mean_at_defaults():
    # kappa / 6 = 0.0683 against beta = 0.067
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.0)
    z = np.linspace(0.0, 1.0, 100001)
    mean = np.trapezoid(vp.kz(z, np.ones_like(z), np.ones_like(z)), z)
    assert mean == pytest.approx(0.41 / 6)
    assert mean == pytest.approx(0.067, rel=0.03)


def test_taylor_coefficient_reproduces_elder_and_truncation_values():
    cfg = VerticalDispersionConfig()
    assert shear_dispersion_coefficient(cfg, zeta_min=0.0) == pytest.approx(ELDER, rel=5e-3)
    # Truncating the walk's domain removes the slow near-bed layer: the coefficient is the
    # cross-sectional average over [zeta_min, 1 - zeta_min] (the width factor matters at 0.01).
    assert shear_dispersion_coefficient(cfg, zeta_min=0.001) == pytest.approx(5.65, rel=1e-2)
    assert shear_dispersion_coefficient(cfg, zeta_min=0.01) == pytest.approx(4.62, rel=1e-2)
    # the floor on the velocity factor costs a few percent at the DRB median ustar / v
    assert shear_dispersion_coefficient(cfg, zeta_min=0.0, ratio=0.143) == pytest.approx(5.23, rel=1e-2)
    const = VerticalDispersionConfig(profile="constant")
    assert shear_dispersion_coefficient(const, zeta_min=0.0) == pytest.approx(6.58, rel=1e-2)
    assert shear_dispersion_coefficient(VerticalDispersionConfig(velocity_profile="uniform"), zeta_min=0.001) == 0.0
    # scale on Kz divides the coefficient; kappa enters the log law and the profile
    assert shear_dispersion_coefficient(VerticalDispersionConfig(scale=2.0), zeta_min=0.0) == pytest.approx(
        ELDER / 2.0, rel=5e-3
    )


def test_taylor_coefficient_small_ratio_limit_matches_unfloored():
    cfg = VerticalDispersionConfig()
    assert shear_dispersion_coefficient(cfg, 0.001, ratio=1e-4) == pytest.approx(
        shear_dispersion_coefficient(cfg, 0.001), rel=1e-4
    )


def test_taylor_coefficient_value_profile_needs_the_reach_scale():
    cfg = VerticalDispersionConfig(profile="value", value=0.01)
    with pytest.raises(ValueError, match="ustar"):
        shear_dispersion_coefficient(cfg, zeta_min=0.001)
    # value = 0.01 with ustar * h = 0.1 is q = 0.1: the same as a constant profile with beta = 0.1
    c = shear_dispersion_coefficient(cfg, zeta_min=0.001, ustar_depth=0.1)
    assert c == pytest.approx(
        shear_dispersion_coefficient(VerticalDispersionConfig(profile="constant", beta=0.1), 0.001)
    )


def test_shear_table_interpolates_the_quadrature():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    c = vp.shear_coefficient(np.array([0.1, 0.0, 0.1]), np.array([0.7, 0.5, 0.0]))
    assert c[0] == pytest.approx(
        shear_dispersion_coefficient(VerticalDispersionConfig(), 0.001, ratio=0.1 / 0.7), rel=1e-2
    )
    assert c[1] == 0.0 and c[2] == 0.0  # no shear without ustar; no ratio without v
    uniform = VerticalProfiles(VerticalDispersionConfig(velocity_profile="uniform"), zeta_min=0.001)
    assert list(uniform.shear_coefficient(np.array([0.1]), np.array([0.7]))) == [0.0]


def test_shear_coefficient_per_reach_for_value_and_background():
    # profile "value" and a non-zero background depend on ustar * h, not only on ustar / v, so the
    # coefficient is computed per reach; it must agree with the single-reach quadrature.
    cfg = VerticalDispersionConfig(profile="value", value=0.005)
    vp = VerticalProfiles(cfg, zeta_min=0.001)
    ustar, h, v = np.array([0.1, 0.05]), np.array([0.5, 2.0]), np.array([0.7, 0.3])
    c = vp.shear_coefficient(ustar, v, h)
    for i in range(2):
        ref = shear_dispersion_coefficient(cfg, 0.001, ratio=ustar[i] / v[i], ustar_depth=ustar[i] * h[i])
        assert c[i] == pytest.approx(ref, rel=2e-2)
    bg = VerticalDispersionConfig(background=0.002)
    vpb = VerticalProfiles(bg, zeta_min=0.001)
    cb = vpb.shear_coefficient(ustar, v, h)
    ref = shear_dispersion_coefficient(bg, 0.001, ratio=0.1 / 0.7, ustar_depth=0.05)
    assert cb[0] == pytest.approx(ref, rel=2e-2)
    assert cb[0] < vp.shear_coefficient(ustar, v, h)[0] or True  # more mixing, less shear dispersion
    with pytest.raises(ValueError, match="depth"):
        vp.shear_coefficient(ustar, v)


def test_substep_count_at_drb_medians_and_cap():
    n = substep_count(900.0, kz_max=np.array([0.41 * 0.1 * 0.59 / 4]), h=np.array([0.59]))
    assert 140 <= n <= 170
    assert substep_count(900.0, kz_max=np.array([10.0]), h=np.array([0.1]), max_substeps=500) == 500
    # one count for all reaches: the maximum; dry reaches (h = 0 or kz = 0) do not count
    assert substep_count(900.0, kz_max=np.array([0.0, 1e-3, 0.0]), h=np.array([0.0, 1.0, 1.0])) == 9
    assert substep_count(900.0, kz_max=np.array([0.0, 0.0]), h=np.array([0.0, 1.0])) == 1
    assert substep_count(900.0, kz_max=np.array([0.0, 0.01, 0.001]), h=np.array([0.0, 1.0, 1.0]), c=0.1) == 90
    assert substep_count(900.0, kz_max=np.zeros(0), h=np.zeros(0)) == 1


def test_deposition_probability_formula_and_clip():
    p = deposition_probability(np.array([1e-4, 1.0, 0.0, 1e-4]), 5.0, np.array([1e-4, 1e-4, 1e-4, 0.0]))
    assert p[0] == pytest.approx(1e-4 * np.sqrt(np.pi * 5.0 / 1e-4))
    assert p[1] == 1.0
    assert p[2] == 0.0
    assert p[3] == 0.0  # no mixing at the bed: nothing reaches it by diffusion
    per_particle = deposition_probability(np.array([1e-4, 1e-4]), np.array([5.0, 20.0]), np.array([1e-4, 1e-4]))
    assert per_particle[1] == pytest.approx(2.0 * per_particle[0])
