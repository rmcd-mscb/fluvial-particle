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
