"""Tests for the vertical profiles, Taylor's shear-dispersion coefficient, sub-steps, and deposition."""

import numpy as np
import pytest

from fluvial_particle.network.config import VerticalDispersionConfig
from fluvial_particle.network.vertical import (
    VerticalProfiles,
    deposition_probability,
    shear_dispersion_coefficient,
    sphere_step,
    substep_count,
    vmf_concentration,
    vmf_cosine,
)


ELDER = 0.404 / 0.41**3  # 5.86: Elder's coefficient for the log law with parabolic mixing


def test_log_law_factor_has_unit_mean_after_clip_floor_and_normalization():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    z = np.linspace(0.0, 1.0, 200001)  # the walk lives on the full column
    for ratio in (0.05, 0.143, 0.5, 1.5):
        f = vp.velocity_factor(z, np.full_like(z, ratio), np.ones_like(z))
        assert f.min() >= 0.0
        assert np.trapezoid(f, z) == pytest.approx(1.0, abs=1e-6)
    # the floor is active at ratio 0.5: the raw factor is negative near the bed
    assert (1.0 + 0.5 * (1.0 + np.log(0.001)) / 0.41) < 0.0
    f = vp.velocity_factor(np.array([0.0, 0.001, 1.0]), np.full(3, 0.5), np.ones(3))
    assert f[0] == 0.0 and f[1] == 0.0
    # clipped: below zeta_min the factor is the zeta_min value, above 1 - zeta_min the surface value
    f = vp.velocity_factor(np.array([0.0, 0.0005, 0.001, 0.9995, 1.0]), np.full(5, 0.05), np.ones(5))
    assert f[0] == f[1] == f[2] and f[3] == f[4]


def test_log_law_factor_shape_and_still_water():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    z = np.array([0.1, 0.5, 0.9])
    f = vp.velocity_factor(z, np.full(3, 0.05), np.ones(3))
    assert f[0] < f[1] < f[2]  # faster near the surface
    assert f[1] == pytest.approx(1.0 + 0.05 * (1.0 + np.log(0.5)) / 0.41, rel=1e-3)  # norm is ~1 at small ratio
    assert list(vp.velocity_factor(z, np.zeros(3), np.zeros(3))) == [1.0, 1.0, 1.0]  # v = 0: factor 1
    uniform = VerticalProfiles(VerticalDispersionConfig(velocity_profile="uniform"), zeta_min=0.001)
    assert list(uniform.velocity_factor(z, np.full(3, 0.5), np.ones(3))) == [1.0, 1.0, 1.0]


def test_parabolic_kz_and_max():
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    z = np.array([0.0, 0.25, 0.5, 1.0])
    u = np.full(4, 0.1)
    h = np.full(4, 2.0)
    np.testing.assert_allclose(vp.kz(z, u, h), 0.41 * 0.1 * 2.0 * z * (1 - z))
    assert vp.kz_max(u, h)[0] == pytest.approx(0.41 * 0.1 * 2.0 / 4)
    scaled = VerticalProfiles(VerticalDispersionConfig(scale=2.0, background=0.01), zeta_min=0.001)
    np.testing.assert_allclose(scaled.kz(z, u, h), 2.0 * 0.41 * 0.1 * 2.0 * z * (1 - z) + 0.01)
    assert scaled.kz_max(u, h)[0] == pytest.approx(2.0 * 0.41 * 0.1 * 2.0 / 4 + 0.01)


def test_constant_and_value_profiles():
    z = np.array([0.1, 0.5, 0.9])
    u = np.full(3, 0.1)
    h = np.full(3, 2.0)
    const = VerticalProfiles(VerticalDispersionConfig(profile="constant"), zeta_min=0.001)
    np.testing.assert_allclose(const.kz(z, u, h), 0.067 * 0.1 * 2.0)
    assert const.kz_max(u, h)[0] == pytest.approx(0.067 * 0.1 * 2.0)
    value = VerticalProfiles(VerticalDispersionConfig(profile="value", value=0.003, background=0.001), zeta_min=0.001)
    np.testing.assert_allclose(value.kz(z, u, h), 0.004)
    assert value.kz_max(u, h)[0] == pytest.approx(0.004)


def test_constant_profile_matches_parabolic_depth_mean_at_defaults():
    # kappa / 6 = 0.0683 against beta = 0.067
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)  # Kz does not depend on the clip
    z = np.linspace(0.0, 1.0, 100001)
    mean = np.trapezoid(vp.kz(z, np.ones_like(z), np.ones_like(z)), z)
    assert mean == pytest.approx(0.41 / 6)
    assert mean == pytest.approx(0.067, rel=0.03)


def test_taylor_coefficient_reproduces_elder_and_clip_values():
    cfg = VerticalDispersionConfig()
    assert shear_dispersion_coefficient(cfg, zeta_min=0.0) == pytest.approx(ELDER, rel=5e-3)
    # Clipping the log law at zeta_min holds the near-bed velocity at its zeta_min value instead of
    # letting it fall to -inf: a small loss of shear dispersion (0.5 percent at 0.001, 5 at 0.01).
    assert shear_dispersion_coefficient(cfg, zeta_min=0.001) == pytest.approx(5.83, rel=1e-2)
    assert shear_dispersion_coefficient(cfg, zeta_min=0.01) == pytest.approx(5.58, rel=1e-2)
    # the floor on the velocity factor costs a few percent at the DRB median ustar / v, and once it
    # is active the clip no longer matters (the floored region includes the clipped one)
    assert shear_dispersion_coefficient(cfg, zeta_min=0.0, ratio=0.143) == pytest.approx(5.23, rel=1e-2)
    assert shear_dispersion_coefficient(cfg, zeta_min=0.001, ratio=0.143) == pytest.approx(
        shear_dispersion_coefficient(cfg, zeta_min=0.0, ratio=0.143), rel=1e-4
    )
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


def test_taylor_coefficient_is_zero_without_vertical_mixing():
    cfg = VerticalDispersionConfig(profile="value", value=0.0)
    assert shear_dispersion_coefficient(cfg, zeta_min=0.001, ustar_depth=0.1) == 0.0
    vp = VerticalProfiles(cfg, zeta_min=0.001)
    assert list(vp.shear_coefficient(np.array([0.1]), np.array([0.7]), np.array([1.0]))) == [0.0]


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
    # more vertical mixing means less shear dispersion: the background lowers c below the table value
    table = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001).shear_coefficient(ustar, v)
    assert cb[0] < table[0]
    with pytest.raises(ValueError, match="depth"):
        vp.shear_coefficient(ustar, v)
    with pytest.raises(ValueError, match="no positive depth"):
        vp.shear_coefficient(ustar, v, np.array([0.5, 0.0]))


def test_vertical_profiles_validates_zeta_min_and_tau():
    with pytest.raises(ValueError, match="zeta_min"):
        VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.7)
    with pytest.raises(ValueError, match="zeta_min"):
        VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.0)
    with pytest.raises(ValueError, match="tau"):
        vmf_concentration(np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="tau"):
        sphere_step(np.array([0.5]), np.array([-1.0]), np.random.RandomState(0))
    with pytest.raises(ValueError, match="ustar_depth"):
        shear_dispersion_coefficient(VerticalDispersionConfig(profile="value", value=0.01), 0.001, ustar_depth=0.0)


def test_shear_table_covers_every_ratio():
    # tabulated in x = ratio / (1 + ratio), so slow steep reaches (ustar / v > 2) are not clamped
    vp = VerticalProfiles(VerticalDispersionConfig(), zeta_min=0.001)
    ratios = np.array([0.073, 0.5, 2.0, 5.0, 10.0])
    c = vp.shear_coefficient(ratios, np.ones(5))
    for r, ci in zip(ratios, c, strict=True):
        ref = shear_dispersion_coefficient(VerticalDispersionConfig(), 0.001, ratio=float(r))
        assert ci == pytest.approx(ref, rel=0.03), (r, ci, ref)
    sweep = vp.shear_coefficient(np.logspace(-1.3, 3, 50), np.ones(50))
    assert np.all(np.diff(sweep) <= 0.0) and sweep[-1] < 1e-3  # tends to 0 as the ratio grows
    # a denormal velocity overflows the ratio; it is a still reach, not a NaN in the correction
    assert vp.shear_coefficient(np.array([1.0]), np.array([5e-324]))[0] == 0.0


def test_vmf_concentration_small_kappa_limit():
    # coth k - 1 / k ~ k / 3 there, so kappa ~ 3 exp(-2 tau); the old evaluation cancelled to noise
    tau = np.array([7.0, 9.0, 11.5])
    np.testing.assert_allclose(vmf_concentration(tau), 3.0 * np.exp(-2.0 * tau), rtol=5e-3)


def test_vmf_table_is_monotone_up_to_tau_max():
    taus = np.logspace(-4, np.log10(11.9), 300)  # inside (TAU_MIN, TAU_MAX)
    kappa = vmf_concentration(taus)
    assert np.all(np.diff(kappa) < 0.0) and kappa[-1] > 0.0


def test_substep_count_at_drb_medians_and_cap():
    kz = np.array([0.41 * 0.1 * 0.59 / 4])
    h = np.array([0.59])
    assert 140 <= substep_count(900.0, kz_max=kz, h=h, c=0.1) <= 170  # the spec's screening number
    assert 450 <= substep_count(900.0, kz_max=kz, h=h) <= 560  # the default fraction 0.03
    assert substep_count(900.0, kz_max=np.array([10.0]), h=np.array([0.1]), max_substeps=500) == 500
    # one count for all reaches: the maximum; dry reaches (h = 0 or kz = 0) do not count
    assert substep_count(900.0, kz_max=np.array([0.0, 1e-3, 0.0]), h=np.array([0.0, 1.0, 1.0]), c=0.1) == 9
    assert substep_count(900.0, kz_max=np.array([0.0, 0.0]), h=np.array([0.0, 1.0])) == 1
    assert substep_count(900.0, kz_max=np.array([0.0, 0.01, 0.001]), h=np.array([0.0, 1.0, 1.0]), c=0.1) == 90
    assert substep_count(900.0, kz_max=np.zeros(0), h=np.zeros(0)) == 1


def test_vmf_calibration_and_cosine_sampler():
    tau = np.array([0.0, 1e-7, 0.01, 0.4, 2.0, 50.0])
    kappa = vmf_concentration(tau)
    assert np.isinf(kappa[0]) and kappa[1] == pytest.approx(0.5 / 1e-7) and kappa[5] == 0.0
    for k, t in zip(kappa[2:5], tau[2:5], strict=True):
        assert (1.0 / np.tanh(k) - 1.0 / k) == pytest.approx(np.exp(-2.0 * t), rel=1e-3)
    u = np.linspace(0.0, 1.0, 100001)
    w = vmf_cosine(np.full_like(u, 5.0), u)
    assert w.min() >= -1.0 and w.max() <= 1.0 and w[0] == 1.0
    # the density is proportional to exp(kappa w): the mean is coth(kappa) - 1 / kappa
    assert np.trapezoid(w, u) == pytest.approx(1.0 / np.tanh(5.0) - 0.2, rel=1e-4)
    assert list(vmf_cosine(np.array([np.inf, 0.0]), np.array([0.3, 0.3]))) == [1.0, 0.4]


def test_sphere_step_keeps_a_uniform_column_uniform_at_any_step():
    from scipy import stats

    # Invariance is exact (a random rotation of a uniform direction is uniform), so the p-values are
    # themselves uniform over seeds; one fixed seed is checked against a 1e-3 threshold.
    rng = np.random.RandomState(0)
    z = rng.uniform(0.0, 1.0, 200000)
    for tau in (0.01, 0.5, 5.0):
        z = sphere_step(z, np.full_like(z, tau), rng)
        assert z.min() >= 0.0 and z.max() <= 1.0
        counts, _ = np.histogram(z, bins=50, range=(0.0, 1.0))
        assert stats.chisquare(counts).pvalue > 1e-3
    # tau = 0 is the identity
    z0 = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(sphere_step(z0, np.zeros(3), rng), z0)
    # first-mode decay: E[1 - 2 zeta'] for particles starting at the surface (cos theta = 1) is exp(-2 tau)
    zs = np.zeros(200000)
    out = sphere_step(zs, np.full_like(zs, 0.3), rng)
    assert np.mean(1.0 - 2.0 * out) == pytest.approx(np.exp(-0.6), abs=3e-3)


def test_mix_dispatches_by_profile_and_keeps_uniformity():
    from scipy import stats

    rng = np.random.RandomState(1)
    n = 100000
    u = np.full(n, 0.1)
    h = np.full(n, 1.0)
    dt = np.full(n, 20.0)
    for cfg in (
        VerticalDispersionConfig(),
        VerticalDispersionConfig(profile="constant"),
        VerticalDispersionConfig(profile="value", value=0.002),
        VerticalDispersionConfig(background=0.001),
    ):
        vp = VerticalProfiles(cfg, 0.001)
        z = rng.uniform(0.0, 1.0, n)
        for _ in range(5):
            z = vp.mix(z, u, h, dt, rng)
        assert z.min() >= 0.0 and z.max() <= 1.0
        counts, _ = np.histogram(z, bins=20, range=(0.0, 1.0))
        assert stats.chisquare(counts).pvalue > 0.01, cfg
    # a dry reach leaves zeta alone
    z = np.array([0.2, 0.7])
    np.testing.assert_array_equal(
        VerticalProfiles(VerticalDispersionConfig(), 0.001).mix(z, np.zeros(2), np.zeros(2), dt[:2], rng), z
    )


def test_deposition_probability_formula_and_clip():
    # p = k_d dt_sub / (zeta_min h + w dt_sub): the Robin flux k_d C_bed out of the particles that
    # make contact per sub-step (those within the layer plus the settling displacement)
    p = deposition_probability(np.array([1e-4, 1.0, 0.0, 1e-4]), 5.0, np.array([1e-3, 1e-3, 1e-3, 0.0]))
    assert p[0] == pytest.approx(1e-4 * 5.0 / 1e-3)
    assert p[1] == 1.0
    assert p[2] == 0.0
    assert p[3] == 0.0  # no layer (a dry reach): nothing deposits
    assert deposition_probability(np.array([1e-3]), 1.0, np.array([0.0]), settling=0.01)[0] == 0.0  # even when settling
    per_particle = deposition_probability(np.array([1e-5, 1e-5]), np.array([5.0, 20.0]), np.array([1e-3, 1e-3]))
    assert per_particle[1] == pytest.approx(4.0 * per_particle[0])
    # settling adds contacts: the settling-limited probability is k_d / w
    p = deposition_probability(np.array([1e-3]), 5.0, np.array([1e-3]), settling=0.05)
    assert p[0] == pytest.approx(1e-3 * 5.0 / (1e-3 + 0.05 * 5.0))
    assert p[0] == pytest.approx(1e-3 / 0.05, rel=1e-2)
    # an upward velocity does not
    assert deposition_probability(np.array([1e-4]), 5.0, np.array([1e-3]), settling=-0.05)[0] == pytest.approx(0.5)
