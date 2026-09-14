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


def _run(tmp_path, ds, dt, end_seconds, output_interval, dispersion, release_s=0.0, particles=N):
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
                "reach_id": 1,
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
