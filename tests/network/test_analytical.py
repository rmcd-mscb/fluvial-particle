"""Analytical acceptance tests on a uniform chain: exact advection, Gaussian plume, inverse-Gaussian arrivals."""

import numpy as np
import pytest
from scipy import stats

from fluvial_particle.network.run import run_network_simulation
from tests.network.support import chain_dataset, write_network_file


pytestmark = pytest.mark.slow

N = 20000
K = 50.0
V = 1.0
L_REACH = 1000.0
N_REACH = 10
S0 = 500.0

# Broadie, Glasserman & Kou, Math. Finance 7(4), 1997, "A continuity correction for discrete
# barrier options": a barrier monitored only at discrete instants behaves like a continuously
# monitored barrier shifted outward by BETA * sigma_step, where sigma_step is the per-step
# standard deviation of the monitored process.
BETA = 0.5826


def _run(tmp_path, ds, dt, end_seconds, output_interval, dispersion, release_s=0.0, particles=N, reach_id=1):
    path = write_network_file(tmp_path / "chain.nc", ds)
    end = np.datetime64("1979-01-01", "ns") + np.timedelta64(int(end_seconds), "s")
    cfg = {
        "hydraulics_file": str(path),
        "dt": dt,
        "output_interval": output_interval,
        "end_time": str(end),
        "dispersion": dispersion,
        "sources": [
            {
                "reach_id": reach_id,
                "form": "slug",
                "time": 0.0,
                "mass": float(particles),
                "particles": particles,
                "s": release_s,
            }
        ],
    }
    return run_network_simulation(cfg, tmp_path / "out", seed=12345, quiet=True)


@pytest.mark.parametrize("dt", [900.0, 86400.0])
def test_pure_advection_exit_times_are_exact(tmp_path, dt):
    velocities = [1.0, 0.5, 2.0, 1.0, 4.0]
    ds = chain_dataset(n_reach=5, length=L_REACH, velocity=velocities)
    expected = sum(L_REACH / v for v in velocities)  # 4750 s
    with _run(
        tmp_path, ds, dt=dt, end_seconds=86400, output_interval=86400.0, dispersion={"model": "none"}, particles=100
    ) as res:
        et = res.arrival_times()["exit_time"].to_numpy()
        assert et.size == 100
        np.testing.assert_allclose(et, expected, rtol=1e-9)


def test_gaussian_plume_moments_and_normality(tmp_path):
    ds = chain_dataset(n_reach=N_REACH, length=L_REACH, velocity=V, k_target=K)
    t_end = 3000.0
    with _run(
        tmp_path,
        ds,
        dt=10.0,
        end_seconds=t_end,
        output_interval=t_end,
        dispersion={"model": "constant", "value": K},
        release_s=S0,
    ) as res:
        df = res.positions(-1)
        assert (df["status"] == 1).all()
        cum_before = np.arange(N_REACH) * L_REACH
        dist = cum_before[df["reach_index"].to_numpy()] + df["s"].to_numpy()
        mean, var = S0 + V * t_end, 2.0 * K * t_end
        assert abs(dist.mean() - mean) < 3.0 * np.sqrt(var / N)
        assert abs(dist.var() - var) < 3.0 * var * np.sqrt(2.0 / N)
        assert stats.normaltest(dist).pvalue > 0.01


def test_mid_chain_release_is_unbiased_at_default_dt(tmp_path):
    """A release just below a reach top disperses without bias under the hybrid upstream rule.

    A particle released mid-chain has no recorded history, so every upstream overshoot in its first
    steps leaves the release reach through its top. Reflecting there (the pre-hybrid rule) acts as a
    hard barrier a few tens of meters above the release point and biases the plume downstream while
    shrinking its variance. Under the hybrid rule the particle instead enters the reach's single
    parent at ``length[parent] + s``, which is exactly continuous in the chain coordinate, so the
    displacement is a plain random walk: mean ``v t`` and variance ``2 K t``.
    """
    v = 0.05  # slow enough that no particle reaches either end of the chain in 10 h
    n_reach, l_reach, s0, t_end = 10, 2000.0, 40.0, 36000.0
    ds = chain_dataset(n_reach=n_reach, length=l_reach, velocity=v, k_target=K)
    with _run(
        tmp_path,
        ds,
        dt=900.0,
        end_seconds=t_end,
        output_interval=t_end,
        dispersion={"model": "constant", "value": K},
        release_s=s0,
        reach_id=5,  # reach index 4
    ) as res:
        df = res.positions(-1)
        assert (df["status"] == 1).all(), "no particle should exit the chain"
        cum_before = np.arange(n_reach) * l_reach
        dist = cum_before[df["reach_index"].to_numpy()] + df["s"].to_numpy()
        disp = dist - (cum_before[4] + s0)
        mean, var = v * t_end, 2.0 * K * t_end
        assert abs(disp.mean() - mean) < 3.0 * np.sqrt(var / N), (
            f"mean displacement {disp.mean():.1f} m vs {mean:.1f} m (3 SE = {3.0 * np.sqrt(var / N):.1f} m)"
        )
        assert abs(disp.var() - var) < 3.0 * var * np.sqrt(2.0 / N), (
            f"displacement variance {disp.var():.4g} vs {var:.4g} (tolerance {3.0 * var * np.sqrt(2.0 / N):.4g})"
        )


def test_inverse_gaussian_arrival_times(tmp_path):
    """Arrival times at the outlet match a continuity-corrected inverse-Gaussian.

    The solver's clock only checks whether a particle has crossed the outlet at the end of each
    step (`NetworkSolver._disperse` stamps `exit_time = t + dt` for a dispersive crossing, with no
    partial-step credit). A discrete-time random walk monitored only at step ends is therefore
    slower to register a crossing than the continuously-monitored process the closed-form inverse
    Gaussian describes: the effective outlet sits `BETA * sigma_step` further downstream, where
    `sigma_step = sqrt(2 K dt)` is the per-step dispersive standard deviation (Broadie, Glasserman
    & Kou 1997), and every recorded exit is additionally stamped `dt / 2` late on average because
    it is rounded up to the end of the step rather than the true (continuous) crossing instant.
    This test corrects the reference distribution's effective length for the first effect and
    recenters the sample for the second, then checks the corrected arrival times against the
    inverse Gaussian implied by the true (uncorrected) travel length and dispersion.

    This two-term correction is exact only when the per-step dispersive kick dominates the
    step's displacement (small dt relative to the travel time), which is the case at dt=10 here.
    At a larger dt where advection dominates the step (dt=900, the project's default), see
    `test_first_passage_bias_bounded_at_default_dt` instead: the two-term expression there is
    only an upper bound on the bias, not an exact correction (most exits occur in the exactly
    monitored advective substep, well before dispersion could carry them past the outlet).
    """
    ds = chain_dataset(n_reach=N_REACH, length=L_REACH, velocity=V, k_target=K)
    length = N_REACH * L_REACH - S0  # 9500 m to the outlet
    dt = 10.0
    with _run(
        tmp_path,
        ds,
        dt=dt,
        end_seconds=20000,
        output_interval=20000.0,
        dispersion={"model": "constant", "value": K},
        release_s=S0,
    ) as res:
        et = res.arrival_times()["exit_time"].to_numpy()
        assert et.size == N, "every particle should have exited"
        t = et - dt / 2.0
        length_eff = length + BETA * np.sqrt(2.0 * K * dt)
        mean = length_eff / V
        shape = length_eff**2 / (2.0 * K)
        dist = stats.invgauss(mean / shape, scale=shape)
        assert abs(t.mean() - mean) < 3.0 * np.sqrt(mean**3 / shape / N)
        assert stats.kstest(t, dist.cdf).pvalue > 0.01


def test_first_passage_bias_bounded_at_default_dt(tmp_path):
    """At dt=900 s (the project's default step), the first-passage bias is small and bounded above.

    The two-term Broadie-Glasserman-Kou correction used in `test_inverse_gaussian_arrival_times`
    (`BETA * sqrt(2 K dt) / v` for the discrete-monitoring shift, plus `dt / 2` for exit-time
    rounding) is exact only when the per-step dispersive kick dominates the step's displacement.
    At dt=900 with v=1 m/s, each step advects 900 m -- three times the ~300 m kick standard
    deviation (`sqrt(2 K dt)` = `sqrt(2*50*900)` ~ 300 m) -- so most particles cross the outlet
    during the exactly-monitored advective substep (`NetworkSolver._advect` stamps
    `exit_time = t + dt - time_left`, exact to the true crossing instant) well before the
    dispersive kick (`_disperse`, `exit_time = t + dt`, no partial-step credit) would apply. The
    two-term expression is therefore only an upper bound on the bias here, not an exact
    correction: this test checks the observed bias falls between 0 (net of sampling noise) and
    that bound, and that the arrival distribution's spread still matches the (uncorrected)
    inverse-Gaussian standard deviation to within 10%, confirming the process is still the same
    inverse-Gaussian first-passage law, just with a much smaller discrete-monitoring bias than
    the worst-case bound predicts.
    """
    ds = chain_dataset(n_reach=N_REACH, length=L_REACH, velocity=V, k_target=K)
    length = N_REACH * L_REACH - S0  # 9500 m to the outlet
    dt = 900.0
    with _run(
        tmp_path,
        ds,
        dt=dt,
        end_seconds=18000,
        output_interval=900.0,
        dispersion={"model": "constant", "value": K},
        release_s=S0,
    ) as res:
        et = res.arrival_times()["exit_time"].to_numpy()
        assert et.size == N, "every particle should have exited"
        mean = length / V  # uncorrected (unshifted) inverse-Gaussian mean and shape
        shape = length**2 / (2.0 * K)
        se = np.sqrt(mean**3 / shape / N)
        bias = et.mean() - mean
        upper_bound = BETA * np.sqrt(2.0 * K * dt) / V + dt / 2.0 + 3.0 * se
        assert -3.0 * se <= bias <= upper_bound
        sample_std = et.std()
        ig_std = np.sqrt(mean**3 / shape)
        assert abs(sample_std - ig_std) / ig_std < 0.10


# ---- behavioral particles: the drift model against the vertical physics -------------------------
#
# All runs build DriftParticles directly on an in-memory uniform reach (ustar 0.1 m/s, depth 1 m,
# velocity 0.7 m/s: near the DRB export medians, depth rounded up from 0.59 m) with 20,000 particles released as a slug at s = 0 of one
# long reach, dt = 60 s, seed 12345. The vertical kernel keeps a uniform column exactly uniform at any
# sub-step; the default sub-step fraction 0.03 over-predicts the within-step shear dispersion by about
# 1 percent (8 percent at the spec's original 0.1), the error of sampling the velocity once per sub-step.

from fluvial_particle.network.config import DispersionConfig, VerticalDispersionConfig  # noqa: E402
from fluvial_particle.network.network import Network  # noqa: E402
from fluvial_particle.network.particles import DriftParticles  # noqa: E402
from fluvial_particle.network.solver import ACTIVE, SETTLED  # noqa: E402
from fluvial_particle.network.sources import ParticleSchedule  # noqa: E402
from fluvial_particle.network.vertical import VerticalProfiles  # noqa: E402
from tests.network.support import ArrayHydraulicsProvider, uniform_reach_dataset  # noqa: E402


USTAR, DEPTH, VEL = 0.1, 1.0, 0.7
T0 = np.datetime64("1979-01-01", "ns")
NO_LONGITUDINAL = DispersionConfig(model="none")


def _drift(n=N, *, dispersion=NO_LONGITUDINAL, params=None, dt=60.0, width=10.0, seed=12345, length=200000.0):
    ds = uniform_reach_dataset(length=length, velocity=VEL, depth=DEPTH, ustar=USTAR, width=width)
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    net = Network(prov.static)
    sch = ParticleSchedule(
        np.zeros(n, dtype=np.int32), np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n, dtype=np.int32)
    )
    return DriftParticles(
        net, prov, sch, start_time=T0, dt=dt, dispersion=dispersion, rng=np.random.RandomState(seed), params=params
    )


def _advance(sol, seconds):
    for _ in range(round(seconds / sol.dt)):
        sol.step()
    return sol


def _uniform_pvalue(zeta, bins=20):
    counts, _ = np.histogram(zeta, bins=bins, range=(0.0, 1.0))
    return stats.chisquare(counts).pvalue


def test_drift_uniform_column_stays_uniform(monkeypatch):
    """Test 1: a passive drift particle column that starts well mixed stays well mixed.

    The negative control replaces the mixing kernel with the Euler-Ito walk of the original design
    (gradient drift term, reflection at zeta_min) at the same sub-steps: the walls of a diffusivity
    that vanishes at the boundary throw particles to mid-column and the column is not uniform.
    """
    sol = _advance(_drift(), 3600.0)
    p = _uniform_pvalue(sol.state["zeta"])
    assert p > 0.01, f"chi-square p = {p:.3g}"
    assert (sol.status == ACTIVE).all()

    from fluvial_particle.random_walk import reflect_interval

    def euler_ito(self, zeta, ustar, h, dt, rng):
        a = self.cfg.kappa * ustar / h
        drift = a * (1.0 - 2.0 * zeta)
        noise = np.sqrt(2.0 * a * zeta * (1.0 - zeta) * dt) * rng.standard_normal(zeta.size)
        return reflect_interval(zeta + drift * dt + noise, 0.001, 0.999)

    monkeypatch.setattr(VerticalProfiles, "mix", euler_ito)
    sol = _advance(_drift(), 3600.0)
    assert _uniform_pvalue(sol.state["zeta"]) < 1e-6


def test_drift_rouse_profile():
    """Test 2: a settling particle column equilibrates to the Rouse profile (P = 0.5).

    ``settling_velocity = P kappa ustar`` with a reflecting bed. The stationary density of the
    continuous process is the Rouse profile ((1 - zeta) / zeta)^P, i.e. Beta(1 - P, 1 + P), which
    is normalizable only for P < 1. The split (mix, shift, reflect) flattens the singular bed layer
    of thickness about a dt_sub each sub-step, so the Kolmogorov-Smirnov distance to the exact
    profile converges as dt_sub^(1 - P), not linearly: about 0.085 at the default fraction 0.03,
    0.05 at 0.01 and 0.03 at 0.003 (measured in the review of this PR). The test runs at 0.01
    against the untruncated Beta CDF with a threshold of 0.06. Rouse numbers of 1 and above are
    out of reach: their profile is not integrable at the bed.
    """
    p_rouse = 0.5
    w = p_rouse * 0.41 * USTAR
    sol = _advance(_drift(params={"settling_velocity": w, "substep_fraction": 0.01}), 3 * 3600.0)
    ks = stats.kstest(sol.state["zeta"], stats.beta(1.0 - p_rouse, 1.0 + p_rouse).cdf)
    assert ks.statistic < 0.06, f"P = {p_rouse}: KS D = {ks.statistic:.4f} (p = {ks.pvalue:.3g})"
    # the Beta(1 - P, 1 + P) mean (1 - P) / 2 is a seed-insensitive check on the settling split's bias
    assert abs(sol.state["zeta"].mean() - 0.5 * (1.0 - p_rouse)) < 0.01


def test_drift_mean_advection_is_the_reach_velocity():
    """Test 3: a well-mixed passive drift column advects at the reach velocity (within 3 SE)."""
    t_end = 6 * 3600.0
    sol = _advance(_drift(), t_end)
    s = sol.s
    assert (sol.status == ACTIVE).all()
    se = s.std() / np.sqrt(s.size)
    assert abs(s.mean() - VEL * t_end) < 3.0 * se, f"mean {s.mean():.2f} vs {VEL * t_end:.2f}, SE {se:.2f}"


@pytest.mark.parametrize("profile", ["parabolic", "constant"])
def test_drift_shear_dispersion_emerges(profile):
    """Test 4: the resolved vertical shear produces Taylor's longitudinal dispersion 2 c ustar h t."""
    t_end = 6 * 3600.0
    vert = VerticalDispersionConfig(profile=profile)
    sol = _advance(_drift(dispersion=DispersionConfig(model="none", vertical=vert)), t_end)
    c = sol.profiles.shear_coefficient(np.array([USTAR]), np.array([VEL]), np.array([DEPTH]))[0]
    expected = 2.0 * c * USTAR * DEPTH * t_end
    ratio = sol.s.var() / expected
    assert abs(ratio - 1.0) < 0.10, f"{profile}: variance / (2 c ustar h t) = {ratio:.3f} with c = {c:.2f}"


def test_drift_shear_correction_restores_fischer():
    """Test 4b: with Fischer on and the correction active, the total variance is 2 K_fischer t.

    The width is chosen so the Fischer coefficient is about 2 m^2/s, of which the resolved shear
    (c ustar h, about 0.5 m^2/s) is a quarter: the correction has to remove a visible share.
    """
    t_end = 6 * 3600.0
    k_target = 2.0
    width = float(np.sqrt(k_target * DEPTH * USTAR / (0.011 * VEL**2)))
    sol = _advance(_drift(dispersion=DispersionConfig(model="fischer", shear_correction="auto"), width=width), t_end)
    h = sol.provider.hydraulics(T0)
    k_fischer = 0.011 * VEL**2 * float(h["width"][0]) ** 2 / (DEPTH * USTAR)
    ratio = sol.s.var() / (2.0 * k_fischer * t_end)
    assert abs(ratio - 1.0) < 0.10, f"variance / (2 K t) = {ratio:.3f}, K = {k_fischer:.2f}"
    off = _advance(_drift(dispersion=DispersionConfig(model="fischer", shear_correction="off"), width=width), t_end)
    assert off.s.var() > sol.s.var() * 1.1  # without the correction the shear part is counted twice


def _robin_reference(k_d, w, kz, h, times, *, n_cells=200, dt=0.5):
    """Deposited fraction from a finite-volume solution of dC/dt = d/dz (Kz dC/dz + w C) on [0, h].

    Cells of width ``dz`` with faces between them; the upward flux at an interior face is
    ``-Kz (C_above - C_below) / dz - w C_above`` (central diffusion, upwind settling), 0 at the
    surface, and ``-k_d C_0`` at the bed: the Robin condition, deposition at ``k_d`` times the bed
    concentration. Backward Euler in time from a uniform column of unit mass.
    """
    from scipy.linalg import solve_banded

    dz = h / n_cells
    d = kz / dz**2
    c = np.full(n_cells, 1.0 / h)
    # dC/dt = L C: interior L[i,i] = -2 d - w/dz, L[i,i+1] = d + w/dz, L[i,i-1] = d;
    # bed cell: L[0,0] = -d - k_d/dz, L[0,1] = d + w/dz; surface cell: L[n-1,n-1] = -d - w/dz
    diag = np.full(n_cells, -2.0 * d - w / dz)
    diag[0] = -d - k_d / dz
    diag[-1] = -d - w / dz
    sup = np.full(n_cells - 1, d + w / dz)  # L[i, i+1]
    sub = np.full(n_cells - 1, d)  # L[i+1, i]
    # backward Euler: (I - dt L) c_new = c_old, in solve_banded's (upper, diagonal, lower) layout
    ab = np.zeros((3, n_cells))
    ab[0, 1:] = -dt * sup
    ab[1, :] = 1.0 - dt * diag
    ab[2, :-1] = -dt * sub
    deposited = []
    t = 0.0
    lost = 0.0
    for t_out in times:
        while t < t_out - 1e-9:
            c = solve_banded((1, 1), ab, c)
            lost += k_d * c[0] * dt
            t += dt
        deposited.append(lost)  # k_d C_0 dt is the mass deposited per unit area; the column holds unit mass
    return np.array(deposited)


@pytest.mark.filterwarnings("ignore:the per-contact deposition probability clipped")
def test_drift_deposition_against_the_robin_condition():
    """Test 7: the deposited fraction follows the Robin bed condition with the deposition velocity.

    Run on the constant mixing profile, where the Robin condition is a regular boundary condition
    (with the parabolic profile Kz vanishes at the bed and the continuum condition degenerates to
    the settling flux). The reference is a finite-volume solution with 200 cells and 0.5 s steps.

    The per-contact probability applies to the particles within ``zeta_min h + w dt_sub`` of the
    bed, so it samples the mean density over that width rather than the bed density: the rate is
    low by about ``w (zeta_min h + w dt_sub) / (2 K)`` relative, the sub-step bias bound stated in
    the docs (3 percent at the default fraction 0.03 for this case, converging as the sub-step
    shrinks: 0.024, 0.014, 0.007 absolute at 30 min for fractions 0.03, 0.01, 0.003). The test runs
    at 0.003 after the initial transient; doubling the sub-step moves the answer by less than
    0.01, which shows the derived probability, not the sub-step count, sets the rate. A huge
    deposition velocity clips the probability at 1, which is a Robin bed with the effective
    velocity ``w + zeta_min h / dt_sub`` rather than an absorbing bed; that case is checked against
    the reference at ``k_eff`` with and without settling.
    """
    w, k_d = 0.01, 1e-3
    vert = VerticalDispersionConfig(profile="constant")
    kz = 0.067 * USTAR * DEPTH
    times = np.array([1200.0, 1800.0])
    reference = _robin_reference(k_d, w, kz, DEPTH, times)
    fraction = 0.003

    def deposited_fraction(params, at=times):
        sol = _drift(dispersion=DispersionConfig(model="none", vertical=vert), params=params, dt=min(60.0, at[0]))
        out = []
        for t_out in at:
            _advance(sol, t_out - sol.time)
            out.append(float((sol.status == SETTLED).mean()))
        return np.array(out), sol

    got, _ = deposited_fraction({"settling_velocity": w, "deposition_velocity": k_d, "substep_fraction": fraction})
    assert np.all(np.abs(got - reference) < 0.02), f"particles {got} vs Robin reference {reference}"
    coarse, _ = deposited_fraction({
        "settling_velocity": w,
        "deposition_velocity": k_d,
        "substep_fraction": 2 * fraction,
    })
    assert np.all(np.abs(coarse - got) < 0.01), f"sub-step doubled: {coarse} vs {got}"
    # A clipped probability is not an absorbing bed: every contact deposits, which is a Robin bed with
    # the effective velocity k_eff = w + zeta_min h / dt_sub, with and without settling. With settling
    # the column empties within minutes, so that case is sampled early enough for the reference to
    # discriminate (at 60 s: 0.56 at k_eff, 0.51 without the layer term, 0.85 for an absorbing bed).
    for settling, at in ((w, np.array([60.0, 120.0, 180.0])), (0.0, times)):
        clipped, sol = deposited_fraction(
            {"settling_velocity": settling, "deposition_velocity": 1e6, "substep_fraction": fraction}, at
        )
        dt_sub = sol.dt / sol.last_substeps
        k_eff = settling + sol.params["zeta_min"] * DEPTH / dt_sub
        reference = _robin_reference(k_eff, settling, kz, DEPTH, at)
        assert np.all(np.abs(clipped - reference) < 0.02), (
            f"clipped, w = {settling}: {clipped} vs k_eff {k_eff:.4g}: {reference}"
        )
