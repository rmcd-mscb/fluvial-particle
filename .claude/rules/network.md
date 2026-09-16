# 1D Network Solver (`fluvial_particle.network`)

Separate from the VTK 2D/3D path. Read the specs first:
`docs/superpowers/specs/2026-09-13-network-particle-solver-design.md`, its behavioral-particles addendum
`docs/superpowers/specs/2026-09-16-network-behavioral-particles-design.md`, and the upstream contract in the
pywatershed repo: `docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md`.

| Module | Purpose |
|---|---|
| `provider.py` | `FileHydraulicsProvider`: validate, subset, stream two time slices, hold/linear |
| `network.py` | `Network` topology and polylines; `NetworkBins` |
| `sources.py` | mass-loading sources -> `ParticleSchedule`; `estimate_particles` |
| `solver.py` | `NetworkSolver`: base class and passive model; hooks `on_release`, `behave`; velocity factor; declared state; terminal statuses |
| `particles.py` | `StateVar`; `PARTICLE_MODELS` registry and `resolve_model`; `DriftParticles` |
| `vertical.py` | velocity factor, mixing profiles and kernels (sphere step, reflected Gaussian), Taylor shear coefficient, sub-steps, deposition probability |
| `../random_walk.py` | `reflect_interval`: the fold shared with `Particles.validate_z` |
| `writer.py` | `network_particles.nc` via h5netcdf (mpio under MPI) |
| `results.py` | positions, bins, concentration, arrivals, VTP |
| `run.py` | `run_network_simulation`; CLI in `cli.py` |

Rules: dict keys and file variables use the export's names; solver never depends on bins or polylines;
models declare state and never touch the writer or results; the walk and the shear quadrature use the
same clipped, floored, renormalized velocity factor on the same domain (the full column); vertical
velocity is positive down; the passive model stays bit-identical (the base hooks draw nothing);
tests use synthetic files from `tests/network/support.py`; the DRB file is not committed
(`FLUVIAL_PARTICLE_DRB_FILE` points at it for the optional test). Sample data: the file named by
`FLUVIAL_PARTICLE_DRB_FILE`; on rmcd's machine it lives under
`~/projects/pywatershed/examples/02a_network_hydraulics_export/`.
