"""Run a network particle simulation end to end."""

from __future__ import annotations

import json
import pathlib
import time as _time
import warnings
from collections.abc import Mapping
from os import getpid
from typing import Any

import numpy as np

from .. import __version__
from .config import NetworkConfig
from .dispersion import dispersion_coefficient
from .network import Network
from .particles import resolve_model
from .provider import FileHydraulicsProvider
from .results import NetworkResults
from .sources import ParticleSchedule, expand_sources
from .writer import OUTPUT_FILENAME, NetworkWriter


def resolve_seed(seed: int | None, comm: Any) -> int:
    """A base seed shared by all ranks: the given one, or one derived from time and pid on rank 0.

    Args:
        seed: an explicit seed, or None to derive one.
        comm: MPI communicator to broadcast the seed to every rank, or None.

    Returns:
        The resolved, non-negative base seed.
    """
    if seed is None:
        seed = int(abs(((_time.time() * 181) * ((getpid() - 83) * 359)) % 104729))
    if comm is not None:
        seed = comm.bcast(int(seed), root=0)
    return int(seed)


def diagnostics_report(
    network: Network,
    provider: FileHydraulicsProvider,
    schedule: ParticleSchedule,
    config: NetworkConfig,
    start: np.datetime64,
    end: np.datetime64,
) -> str:
    """Human-readable startup summary: network, sources, dt consequences, memory.

    Args:
        network: the run's network topology.
        provider: the hydraulics provider.
        schedule: the expanded particle release schedule.
        config: the run's configuration.
        start: run start time.
        end: run end time.

    Returns:
        A multi-line diagnostics report.
    """
    lines = [
        "Network particle simulation",
        f"  reaches: {network.n_reach}, outlets: {network.outlets().size}, headwaters: {network.headwaters().size}, "
        f"total length: {network.length.sum() / 1000.0:.1f} km",
        f"  hydraulics: {provider.path} ({provider.times[0]} .. {provider.times[-1]}, {provider.times.size} steps, "
        f"{provider.interpolation})",
        f"  run: {start} .. {end}, dt = {config.dt} s, output every {config.output_interval} s",
        f"  sources: {len(config.sources)} rows, {schedule.n} particles, total mass {schedule.mass.sum():.6g} "
        f"{config.mass_units}",
    ]
    pm = "per-source" if config.particle_mass is None else f"{config.particle_mass:.6g} {config.mass_units}"
    lines.append(f"  particle mass: {pm}")
    n_sources = len(config.sources)
    counts = np.bincount(schedule.source_index, minlength=n_sources)
    mass_per_source = np.bincount(schedule.source_index, weights=schedule.mass, minlength=n_sources)
    for i, row in enumerate(config.sources):
        lines.append(
            f"    source {i}: reach {row['reach_id']} ({row['form']}), {int(counts[i])} particles, "
            f"mass {mass_per_source[i]:.6g} {config.mass_units}"
        )
    inside = provider.times[(provider.times >= start) & (provider.times <= end)]
    # Always anchor the sample to the run window's own endpoints (which hydraulics() accepts under
    # provider.time_window) rather than a timestamp outside it, so a run window that contains no
    # hydraulics timestamp (e.g. a sub-day run on a daily file) still samples successfully.
    candidates = np.unique(np.concatenate([inside, np.array([start, end])]))
    if candidates.size > 64:
        candidates = candidates[np.unique(np.linspace(0, candidates.size - 1, 64).astype(int))]
    sample = candidates
    kicks: list[np.ndarray] = []
    crossed: list[np.ndarray] = []
    d = config.dispersion
    for t in sample:
        h = provider.hydraulics(t)
        wet = np.asarray(h["flow_out"]) > 0.0
        k = dispersion_coefficient(h, d.model, scale=d.scale, cap=d.cap, value=d.value)
        kicks.append(np.sqrt(2.0 * k[wet] * config.dt))
        crossed.append(np.asarray(h["velocity"])[wet] * config.dt > network.length[wet])
    kick = np.concatenate(kicks) if kicks else np.zeros(0)
    cross = np.concatenate(crossed) if crossed else np.zeros(0, dtype=bool)
    med_len = float(np.median(network.length))
    if kick.size:
        lines.append(
            f"  dt check: dispersion kick median {np.median(kick):.0f} m, p95 {np.percentile(kick, 95):.0f} m "
            f"vs median reach length {med_len:.0f} m; {100.0 * cross.mean():.1f}% of wet reach-days crossed in one step"
        )
    est = provider.memory_estimate(schedule.n)
    lines.append(
        f"  memory: window {est['window_bytes'] / 1e6:.1f} MB, static {est['static_bytes'] / 1e6:.1f} MB, "
        f"particles {est['particle_bytes'] / 1e6:.1f} MB, per output time {est['output_time_bytes'] / 1e6:.1f} MB"
    )
    return "\n".join(lines)


def run_network_simulation(
    config: NetworkConfig | Mapping[str, Any] | str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    seed: int | None = None,
    comm: Any = None,
    quiet: bool = False,
) -> NetworkResults | None:
    """Run a network particle simulation and return the results (rank 0; other ranks return None).

    Args:
        config: NetworkConfig, a dict for NetworkConfig.from_dict, or a TOML path.
        output_dir: directory for ``network_particles.nc`` (created if missing).
        seed: base random seed; overrides config.seed. Rank r draws from seed + 1 + r.
        comm: MPI communicator for parallel runs.
        quiet: suppress the startup report.

    Returns:
        NetworkResults on rank 0, None elsewhere.
    """
    cfg = NetworkConfig.coerce(config)
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1
    out = pathlib.Path(output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    with FileHydraulicsProvider(
        cfg.hydraulics_file, interpolation=cfg.interpolation, dtype=cfg.dtype, reach_subset=cfg.reach_subset
    ) as provider:
        start, end = cfg.resolve_times(provider.times)
        provider.time_window = (start, end)
        network = Network(provider.static, crs_wkt=provider.crs_wkt)
        base_seed = resolve_seed(seed if seed is not None else cfg.seed, comm)
        schedule = expand_sources(
            cfg.sources,
            network,
            provider,
            start_time=start,
            end_time=end,
            particle_mass=cfg.particle_mass,
            rng=np.random.RandomState(base_seed),
        )
        n = schedule.n
        lo, hi = rank * n // size, (rank + 1) * n // size
        model_cls = resolve_model(cfg.particles.model)
        solver = model_cls(
            network,
            provider,
            schedule.slice(lo, hi),
            start_time=start,
            dt=cfg.dt,
            dispersion=cfg.dispersion,
            rng=np.random.RandomState(base_seed + 1 + rank),
            max_hops=cfg.max_hops,
            params=cfg.particles.params,
        )
        if rank == 0 and not quiet:
            print(diagnostics_report(network, provider, schedule, cfg, start, end), flush=True)

        total = float((end - start) / np.timedelta64(1, "s"))
        n_steps = int(np.floor(total / cfg.dt + 1e-9))
        every = round(cfg.output_interval / cfg.dt)
        stop = start + np.timedelta64(round(n_steps * cfg.dt * 1e9), "ns")
        if abs(total - n_steps * cfg.dt) > 1e-6:
            warnings.warn(
                f"run window {total:.6g} s is not an integer number of dt={cfg.dt} s steps; "
                f"the simulation stops at {stop} rather than end_time {end}",
                UserWarning,
                stacklevel=2,
            )

        attrs: dict[str, Any] = {
            "hydraulics_file": str(pathlib.Path(cfg.hydraulics_file).resolve()),
            "reach_subset": json.dumps(cfg.to_dict()["reach_subset"]),
            "interpolation": cfg.interpolation,
            "dtype": cfg.dtype,
            "dt": cfg.dt,
            "output_interval": cfg.output_interval,
            "end_time": str(stop.astype("datetime64[s]")),
            "seed": base_seed,
            "mass_units": cfg.mass_units,
            "dispersion": json.dumps(cfg.dispersion.to_dict()),
            "particle_model": cfg.particles.model,
            "particles": json.dumps(cfg.particles.to_dict()),
            "sources": json.dumps(cfg.to_dict()["sources"]),
            "fluvial_particle_version": __version__,
            "created": str(np.datetime64("now", "s")),
            "conventions_note": provider.conventions_note,
        }
        with NetworkWriter(
            out / OUTPUT_FILENAME,
            n_particles=n,
            reach_id=network.reach_id,
            start_time=start,
            attrs=attrs,
            dtype=provider.dtype,
            comm=comm,
            state_specs=solver.state_specs,
        ) as writer:
            writer.write_schedule(schedule.slice(lo, hi), lo, hi)
            writer.write_step(0, 0.0, solver.reach, solver.s, solver.status, lo, hi, state=solver.state)
            itime = 1
            for k in range(n_steps):
                solver.step()
                if (k + 1) % every == 0 or k == n_steps - 1:
                    writer.write_step(
                        itime, solver.time, solver.reach, solver.s, solver.status, lo, hi, state=solver.state
                    )
                    itime += 1
            writer.write_exits(solver.exit_time, solver.exit_reach, lo, hi)
        if rank == 0 and not quiet:
            exited = int((solver.status == 2).sum())
            print(
                f"Done: {n_steps} steps, {itime} output times, {exited} of {solver.n} local particles exited",
                flush=True,
            )
        return NetworkResults(out) if rank == 0 else None
