# 1D river-network passive particle solver

Date: 2026-09-13
Status: draft for review
Branch: `feature/network-solver-spec` (off `main`)

## Purpose

Add a 1D river-network particle tracker to fluvial-particle that transports
passive, non-reacting particles along a reach network using the
"network hydraulics" NetCDF export written by
`pywatershed.utils.export_network_hydraulics`. The first target is a
proof-of-concept demo on the Delaware River Basin (DRB): release particles
at the headwaters and at chosen reaches, track them to the outlets, and
produce arrival-time distributions, concentration along the network, and
a map animation.

The upstream contract, rationale, and screening facts are in the pywatershed
spec `docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md`
(pywatershed repository, branch `feat_network_hydraulics_export`, draft PR
DOI-USGS/pywatershed#420). This spec does not restate the schema beyond what
the solver depends on.

## Non-goals

- Larval, settling, or any behavioral particles. In 1D a first-order
  retention term is settling velocity over depth; the exported depth makes
  that a later addition to the solver's per-step update, not a schema change.
- Reactions, decay, or temperature dependence. `water_temperature` is
  loaded when present and passed through so a later variant can use it.
- Distributaries or braided channels. `to_index` is single-valued in both
  PRMS and NWM sources; the schema cannot express a split. See "Extension
  points".
- An in-memory (BMI-style) provider beyond the test double. The interface
  is designed for it; the implementation is follow-on work.
- An NWM transformer (pywatershed side).
- Any change to the existing 2D/3D VTK solver.

## Facts that shaped the design (DRB export, 2026-09-13)

- 456 reaches, 6 outlets, 114 headwaters, tree topology, 182 daily time
  steps (1979-01-01 to 1979-07-01), 52,792 polyline vertices in EPSG:5070
  meters, 4 polylines not connected to their downstream reach within 1 m.
  Total network length 4,798 km; reach lengths 47 m to 36.6 km, median
  8.6 km.
- Main outlet is reach_id 4205 (351 m^3/s on the last day, ten times the
  next largest outlet); this is the Trenton reach for the demo.
- Masked on `flow_out > 0` (99.8 percent of reach-days): velocity median
  0.70 m/s, max 4.59 m/s; depth median 0.59 m; width median 12 m, max
  196 m; ustar median 0.10 m/s. Residence time median 3.7 h, 98th
  percentile 23 h.
- Fischer coefficient K = 0.011 v^2 w^2 / (d u*): median 13 m^2/s, 95th
  percentile 700 m^2/s, max 2,250 m^2/s. Reach Peclet number v L / K has
  median about 330: transport is advection dominated.
- With a 1 h step, 34 percent of reaches can be crossed within a single
  step on at least one day (max v * 3600 / L is 214). Multi-reach hops
  within one step are routine, not an edge case.
- The DRB file is stored contiguously (no HDF5 chunking). NWM v3 CONUS
  would be 2.7 million reaches at hourly resolution, about 170 MB per time
  slice for eight float64 fields; a year does not fit in memory.

## Decisions

Rulings made during brainstorming, recorded so the plan does not relitigate
them.

1. **Separate subpackage** `fluvial_particle.network` with minimal coupling
   to the VTK code. Shared: TOML/dict configuration style, the `run_*`
   entry-point pattern, the results-object pattern, the RNG helper, and the
   existing VTP/PVD writers for optional ParaView export.
2. **Time step** is user-set (default 900 s). Advection is exact for any dt
   under the export's piecewise-constant convention; dt serves dispersion
   resolution, output resolution, and visualization. A startup diagnostic
   reports the consequences of the chosen dt.
3. **Dispersion**: Fischer (1975) coefficient from the exported hydraulics
   with a user scale factor and optional cap.
4. **Upstream overshoot** from a dispersive kick returns the particle to
   the reach it came from (one level of history); with no history it
   reflects at s = 0.
5. **Daily fields**: linear interpolation between timestamps by default,
   hold (piecewise constant) as an option.
6. **Sources** are mass loadings in time (slug, constant or curve loading,
   or concentration curve converted with the reach flow); particles are
   equal-mass samples of the loading unless a source overrides its count.
7. **Output** is one NetCDF4 file written through h5netcdf on h5py, serial
   or with the mpio driver under MPI, holding only what the solver
   computes. Map positions, per-bin counts, and concentration are
   post-processing in `NetworkResults`.
8. **Concentration** is computed on nearly uniform sub-reach bins of a
   user bin length, on demand, from particle mass and the reach hydraulics
   at that time.
9. **Advection and dispersion are split** within a step: exact advection
   with time carry across reach boundaries, then one dispersive kick with
   displacement carry. This makes the per-step random variance exactly
   2 K dt and lets a uniform chain of reaches reproduce the analytical
   advection-dispersion solutions.
10. **Streaming provider**: two time slices in memory at any time, float32
    option, reach subsetting, polylines loaded on demand.

## Package layout

New subpackage `src/fluvial_particle/network/`:

| Module | Responsibility |
|---|---|
| `provider.py` | `HydraulicsProvider` protocol; `FileHydraulicsProvider` reading the export with xarray (engine h5netcdf), validating it, streaming time slices, interpolating in time |
| `network.py` | `Network`: static topology and geometry, `headwaters()`, `outlets()`, `upstream_of(reach_id)`, `map_position(reach, s)`; `NetworkBins` for sub-reach discretization |
| `sources.py` | Source rows (slug, loading, concentration), expansion into a per-particle schedule, `estimate_particles()` budget helper |
| `dispersion.py` | `fischer_coefficient(velocity, depth, width, ustar, scale, cap)` |
| `solver.py` | `NetworkSolver`: particle state arrays and `step()` |
| `writer.py` | `NetworkWriter`: h5netcdf output, serial or mpio |
| `config.py` | `NetworkConfig` dataclass; from dict or the `[network]` table of a TOML file |
| `results.py` | `NetworkResults`: lazy reader; positions, map positions, counts, concentration, arrival times, DataFrame and VTP export |
| `run.py` | `run_network_simulation()` orchestration |
| `__init__.py` | Public exports |

Outside the subpackage: `pyproject.toml` (dependencies, console scripts),
`cli.py` (two network entry points), the package `__init__.py` (exports),
`docs/` (a network page, options reference, API), `.claude/rules/network.md`.
Nothing in `RiverGrid`, `Particles`, `simulation.py`, `Settings`, or
`results.py` changes.

Each module has one purpose and is testable with in-memory inputs. The
provider is the only module that reads the hydraulics file; the writer is
the only module that writes the particle file; the solver knows nothing
about files.

## Provider

### Protocol

```python
class HydraulicsProvider(Protocol):
    times: np.ndarray                # datetime64[ns], strictly increasing
    static: dict[str, np.ndarray]    # per-reach arrays, see below
    dtype: np.dtype                  # float dtype of the time-varying fields

    def hydraulics(self, t: np.datetime64) -> dict[str, np.ndarray]:
        """Per-reach 'velocity', 'depth', 'width', 'ustar', 'flow_in',
        'flow_out' (and 'water_temperature' when available) at time t."""

    def close(self) -> None: ...
```

`static` carries `reach_id`, `to_index`, `is_outlet`, `length`, `slope`,
`mann_n`, `elevation_mid`, `bankfull_width`, `bankfull_depth`, `x_mid`,
`y_mid`, and, when the file has a polyline block, `vertex_x`, `vertex_y`,
`vertex_dist`, `reach_vertex_start`, `reach_vertex_count`, plus the
`crs_wkt` string under key `crs_wkt`. Dict keys are the export's variable
names, unchanged; they are the coupling vocabulary a future in-memory
provider and BMI facade will share. Consumers must treat `static` as
read-only.

### FileHydraulicsProvider

```python
FileHydraulicsProvider(
    path, *, interpolation="linear", dtype="float64",
    reach_subset=None, time_window=None,
)
```

Validation at open, all failures `ValueError` naming the variable:

- Required variables present with the expected dimensions: static
  (`reach`) `reach_id`, `to_index`, `is_outlet`, `length`, `slope`;
  time-varying (`time`, `reach`) `flow_in`, `flow_out`, `velocity`,
  `depth`, `width`, `ustar`. `units` attributes must equal the schema's
  strings; a missing `units` warns.
- `to_index` is integer, each value -1 or in `[0, nreach)`, and following
  it from every reach reaches -1 within `nreach` steps (a cycle names the
  chain). `is_outlet` agrees with `to_index == -1`.
- When the polyline block is present, `reach_vertex_start + count` lies
  within the `vertex` dimension and `vertex_dist` is non-decreasing within
  each reach.
- `time` is strictly increasing datetime64.

Reach subsetting: `reach_subset` is a sequence of `reach_id` values or
`{"outlet": reach_id}`. The outlet form takes the upstream closure of that
reach via the inverse of `to_index`. The provider keeps only the selected
reaches, remaps `to_index` into the subset (a downstream reach outside the
subset becomes -1 and that reach is treated as an outlet), and applies the
same selection to every read. `static["reach_id"]` is the subset's ids in
file order.

Streaming: time-varying fields are never loaded whole. The provider holds a
window of two time slices, indices `k` and `k+1`, bracketing the last
requested time. `hydraulics(t)` with `times[k+1] <= t` advances the window
by reading slice `k+2` through xarray's lazy indexing (one HDF5 read per
field). Requests are expected to be non-decreasing in `t`; a request before the
current window simply resets it at the cost of two reads, so
`NetworkResults` can sample arbitrary times without ceremony. `time_window=(start, end)` restricts the valid range and
is what `run_network_simulation` passes from the config. A request outside
`[times[0], times[-1]]` (or outside `time_window`) raises.

Interpolation. `linear` returns each field interpolated between slices `k`
and `k+1` by the fraction of the interval elapsed; where `flow_out` is 0 at
either bracketing slice, `velocity` and `ustar` for that reach are set to 0
at any `t` in the interval, so that a dry day never contributes motion. The
other fields interpolate as-is. `hold` returns slice `k` unchanged.

Dtype. `dtype="float32"` casts every time-varying field on read; solver
arrays follow the provider's dtype. Static arrays stay float64 (lengths
need the precision for cumulative arithmetic).

Polylines. The block is read only when `Network.map_position` first needs
it, and only for the reach subset. `static` exposes the arrays through a
lazily populated mapping so `static["vertex_x"]` triggers the read.

Chunking check. At open the provider inspects the HDF5 chunk shape of
`velocity`. If a single time slice spans more than one chunk along `reach`,
or the time chunk length exceeds 32, it warns that streaming will be slow
and suggests rechunking (the DRB file is contiguous, which reads a time
slice as one row and is fine). This is a hint for the NWM transformer on
the pywatershed side to write `(1, nreach)` or similar chunks.

Memory report. `FileHydraulicsProvider.memory_estimate()` returns bytes for
the two-slice window, the static arrays, and per particle (used by the run
diagnostic together with N).

`water_temperature` is served when present, otherwise absent from the
dict.

### In-memory provider

Not implemented in this pass beyond `tests/network/support.py`, where a
minimal `ArrayHydraulicsProvider(times, static, fields)` satisfies the
protocol from arrays built in the test. Its existence in the test suite is
the proof that the interface suffices for a driver that sets arrays each
step. The production version, and a BMI facade around the solver's
initialize / update(dt) / finalize, are follow-on work.

## Network

`Network(static)` wraps the provider's static arrays:

- `n_reach`, `reach_id`, `to_index`, `length`, `is_outlet`.
- `index_of(reach_id)` and `id_of(index)`; unknown ids raise `KeyError`.
- `headwaters()`: indices with no upstream reach (not present in
  `to_index`). Returns reach ids by default, indices with `as_index=True`.
  Same for `outlets()`.
- `upstream_of(reach_id)`: ids of all reaches draining to it, inclusive.
- `parents(index)`: the inverse adjacency, built once as a CSR-style pair
  of arrays; used by `upstream_of` and by the tests, not by the solver
  (the solver's upstream hop uses per-particle history, not topology).
- `map_position(reach, s)`: vectorized over arrays; scales `s / length`
  onto the reach's total `vertex_dist` and interpolates between vertices;
  returns `(x, y)` arrays, NaN where the file has no polylines or the reach
  has fewer than two vertices.
- `crs_wkt`.

`NetworkBins(network, bin_length)` discretizes every reach into
`n = ceil(length / bin_length)` bins of width `length / n`:

- `n_bins`, `bin_reach` (index), `s_start`, `s_end`, `bin_length` (per bin),
  `midpoint_xy` (via `map_position`), `reach_bin_start` (first bin of each
  reach).
- `bin_of(reach, s)`: vectorized lookup returning the bin index.
- `bin_length=np.inf` gives one bin per reach.

## Sources

Every source is a mass loading in time at a point `(reach_id, s)`. Each
source row is a dict (or a TOML table in the `sources` list) with
`reach_id`, at most one of `s` (meters from the upstream end) or `s_frac`
(fraction of the reach length, in [0, 1]; neither means `s = 0`), and one
loading form:

| Form | Fields | Meaning |
|---|---|---|
| `slug` | `time`, `mass` | instantaneous release of `mass` at `time` |
| `loading` | `rate`, `start`, `end`; or `curve` | mass rate: constant `rate` on `[start, end)`, or a table of `(time, rate)` pairs, linear between points, zero outside |
| `concentration` | `value`, `start`, `end`; or `curve` | concentration C(t) at the release point, converted to a mass rate as C(t) times the flow at `(reach, s, t)` |

Times are seconds from the run's `start_time`, or ISO-8601 datetime
strings. Units are the user's: `mass_units` (config, default `"kg"`) labels
`mass` and the rate as `<mass_units>` and `<mass_units> s-1`; a
concentration-form source is in `<mass_units> m-3` (a value in mg/L is
entered as g/m^3 with `mass_units = "g"`). Multiple rows at any reaches and
times are concatenated.

Flow at the release point for the concentration form is
`flow_in + (flow_out - flow_in) * s / length` evaluated at the provider's
interpolation of the bracketing slices; the integral is taken on the union
of the curve's breakpoints and the provider's timestamps within the run.

Expansion happens once at solver initialization, before any particle
arrays exist, and yields per-particle `release_reach` (index), `release_s`,
`release_time` (seconds), `mass`, and `source_index`:

- Total mass per source: the slug mass, or the trapezoid integral of the
  rate over the run window (exact for piecewise-linear curves).
- Particle mass and count: a global `particle_mass` gives every particle
  the same mass and each source `count = round(M / particle_mass)` (at
  least 1 when `M > 0`); a per-source `particles` overrides with a fixed
  count and mass `M / particles`. A source with neither, and no global
  value, is a config error. Mass is exactly conserved per source: the last
  particle absorbs the rounding remainder.
- Release times: at equal-mass quantiles of the cumulative loading,
  `M(t_k) = (k - 0.5) * m` for `k = 1..count` (`spacing = "even"`,
  default), or by inverse-CDF sampling of `M(t) / M` with the run's RNG
  (`spacing = "poisson"`, in which case count is drawn as Poisson(M / m)
  before sampling). A slug releases all its particles at `time`.
- A source whose release times all fall outside `[start_time, end_time]`
  is a config error; one that is partly outside is truncated with a
  warning.

`estimate_particles(config, provider, target_per_bin, bin_length,
reference_travel_time=86400.0)` reports, per source and in total, the
particle mass and count needed for about `target_per_bin` particles in a
bin of `bin_length`: for a continuous loading, `m = rate * bin_length /
(velocity * target)` using the release reach's mean velocity over the
source window; for a slug, the count that puts `target` particles in the
peak bin after `reference_travel_time`, `N = target * sigma * sqrt(2 pi) /
bin_length` with `sigma = sqrt(2 K t)` from the release reach. It returns a
DataFrame and never changes the config; the user decides.

## Solver

### State

Per particle, arrays of length N (the rank's slice under MPI):

| Array | dtype | Meaning |
|---|---|---|
| `reach` | int32 | current reach index, -1 when unreleased or exited |
| `s` | float | distance from the reach's upstream end, NaN when not active |
| `prev_reach` | int32 | the reach the particle most recently arrived from, -1 if none |
| `status` | int8 | 0 unreleased, 1 active, 2 exited |
| `mass` | float | from the source expansion, constant |
| `release_reach`, `release_s`, `release_time` | int32, float, float | from the source expansion |
| `exit_time` | float | seconds from start, NaN until exit |
| `exit_reach` | int32 | outlet reach index, -1 until exit |

`time` is a float clock in seconds from `start_time`. Fields for the step
are sampled from the provider at the step midpoint, `start_time + time +
dt / 2`.

### Step

`step(dt)` advances the clock from `t` to `t + dt`.

1. **Release.** Particles with `release_time` in `(t, t + dt]` become
   active at `(release_reach, release_s)` with `prev_reach = -1`. Their
   time budget for this step is `tau = t + dt - release_time`; every other
   active particle has `tau = dt`. A particle with `release_time <= t`
   that is still unreleased (only possible at the first step when
   `release_time == 0`) is released with `tau = dt`.

2. **Fields.** `h = provider.hydraulics(midpoint)`;
   `K = fischer_coefficient(h, scale, cap)`; `v = h["velocity"]`. Where
   `flow_out == 0`, `v = 0` and `K = 0` (the provider already zeroes
   velocity and ustar; the coefficient function returns 0 where `ustar` or
   `depth` is 0 to avoid division by zero).

3. **Advect with time carry.** `time_left = tau`;
   `s += v[reach] * time_left`. Then loop while any active particle has
   `s > length[reach]`:
   - For those particles: `time_left = (s - length[reach]) / v[reach]`
     (the time remaining after reaching the reach end; `v > 0` is
     guaranteed because `s` only increases when `v > 0`);
     `prev_reach = reach`; `reach = to_index[reach]`.
   - If the new `reach` is -1: `status = 2`,
     `exit_time = t + dt - time_left`, `exit_reach = prev_reach`, `s = NaN`,
     and the particle leaves the loop.
   - Otherwise `s = v[reach] * time_left`. A zero-velocity reach leaves
     `s = 0` and the leftover time is discarded (the particle waits).
   The loop cannot run more than `nreach` times for a valid topology; a
   configurable cap (`max_hops`, default 1000) raises `RuntimeError` if
   exceeded.

4. **Disperse with displacement carry.** For each active particle draw
   `xi ~ N(0, 1)` and `s += xi * sqrt(2 * K[reach] * tau)`, with `K` taken
   from the reach the particle occupies after step 3. Then loop while any
   active particle has `s > length[reach]` or `s < 0`:
   - Downstream (`s > length`): `over = s - length[reach]`;
     `prev_reach = reach`; `reach = to_index[reach]`. If -1: exit with
     `exit_time = t + dt`, `exit_reach = prev_reach`. Otherwise
     `s = over`.
   - Upstream (`s < 0`): if `prev_reach >= 0`, `r = prev_reach`;
     `s = length[r] + s`; `reach = r`; `prev_reach = -1` (one level of
     history; a further upstream overshoot then reflects). If
     `prev_reach < 0`, reflect: `s = -s`.
   Same `max_hops` cap. The kick is drawn once per particle per step; the
   loop only redistributes it. Per-step random variance is therefore
   exactly `2 K tau`, and on a chain of reaches with uniform `K` the sum
   over any number of steps and crossings is exactly `2 K t`.

5. **Clock.** `time += dt`.

Notes on the approximations, first order in dt except the exit-time bias,
which scales as sqrt(dt), and documented in the user docs:

- `K` is sampled from the reach where the particle ends its advective
  move, not time-weighted over the reaches it visited in the step.
- A plain random walk across a jump in `K` slightly over-populates the
  low-`K` side (no drift correction). With reach Peclet numbers of
  hundreds this is below anything the demo can resolve; the generalized
  random walk of LaBolle et al. is the follow-on if it ever matters.
- Exit time from a dispersive hop is the end of the step, and the outlet
  is checked only at step ends, so first-passage times are late by at most
  about 0.5826·sqrt(2 K dt)/v + dt/2 (the Broadie–Glasserman–Kou continuity
  correction plus stamping); the bound is tight when the kick dominates the
  step (sqrt(2 K dt) >> v dt) and the bias is much smaller when advection
  dominates, because most exits then occur in the exactly monitored
  advective substep (about 30 s at dt = 900 s in the acceptance test). This
  term scales as sqrt(dt), not dt, and the analytical arrival-time tests
  cover both regimes.

### Dispersion coefficient

`fischer_coefficient(velocity, depth, width, ustar, scale=1.0, cap=None)`
returns `scale * 0.011 * velocity**2 * width**2 / (depth * ustar)`, 0 where
`depth`, `ustar`, or `velocity` is 0, and `min(K, cap)` when `cap` is set.
A `model = "constant"` option with `value` in m^2/s and `model = "none"`
are provided because they cost two lines and make the analytical tests and
pure-advection checks direct; `"fischer"` is the default.

### Diagnostics at startup

Printed by `run_network_simulation` (rank 0) before the loop:

- Network summary: reaches, outlets, headwaters, total length, time range
  of the file and of the run.
- Sources: rows, total mass, particle mass, N per source and total.
- dt check over the run's time window: median and 95th-percentile
  dispersion kick `sqrt(2 K dt)` versus median reach length; fraction of
  reach-days crossable in one step (`v * dt > length`).
- Memory: provider window, static arrays, particle arrays for N, and the
  per-output-time write size.

### MPI

Particles are embarrassingly parallel. Under `comm`, rank `r` of `size`
owns the contiguous particle slice `[r * N // size, (r + 1) * N // size)`
of the expanded schedule; every rank holds the full provider and network
(they are small relative to particles for any case that needs MPI). No
communication occurs inside `step`. The writer uses the mpio driver;
per-particle static arrays are written by rank 0 before the loop, and each
rank writes its slice at each output time. Per-reach counts are not
computed in the loop, so no reductions are needed. The serial path is the
same code with `comm=None`.

## Output

`NetworkWriter(path, n_particles, n_reach, comm=None)` writes
`network_particles.nc`, a NetCDF4 file, with h5netcdf over h5py (driver
`"mpio"` when `comm` is given, requiring an MPI-enabled h5py).

| Variable | Dims | dtype | Notes |
|---|---|---|---|
| `time` | time (unlimited) | datetime64 | output timestamps |
| `time_seconds` | time | float64 | seconds from `start_time` |
| `reach_index` | (time, particle) | int32 | -1 when not active |
| `s` | (time, particle) | provider dtype | NaN when not active |
| `status` | (time, particle) | int8 | 0 unreleased, 1 active, 2 exited |
| `mass` | particle | float64 | |
| `source_index` | particle | int32 | row in the source table |
| `release_reach` | particle | int32 | index |
| `release_s` | particle | float64 | |
| `release_time` | particle | float64 | seconds from start |
| `exit_time` | particle | float64 | seconds, NaN until exit; written at the end of the run (rewritten by rank slices) |
| `exit_reach` | particle | int32 | -1 until exit |
| `reach_id` | reach | int64 | subset ids, so results can label without the input file |

Global attributes: `hydraulics_file` (absolute path), `reach_subset`
(JSON or null), `interpolation`, `dt`, `output_interval`, `start_time`,
`end_time`, `seed`, `mass_units`, `dispersion` (JSON of the dispersion
settings), `sources` (JSON of the source table), `fluvial_particle_version`,
`created`, `conventions_note` (copied from the input file). Output is
written every `output_interval` seconds, which must be a positive integer
multiple of `dt`; time 0 is always written. `(time, particle)` variables
are chunked `(1, min(N, 2**18))` for time-slice reads and appended one
output time at a time.

Map positions, per-bin counts, and concentration are not written by the
solver.

## Post-processing: NetworkResults

`NetworkResults(output_dir_or_file)` opens the particle file lazily with
xarray (engine h5netcdf), and reopens the hydraulics file named in
`hydraulics_file` (with the same `reach_subset` and `interpolation`) the
first time a method needs hydraulics or polylines. Context-manager and
`close()` as in `SimulationResults`.

- `times`, `n_particles`, `n_reach`, `reach_id`, `network` (a `Network`),
  `sources` (DataFrame from the attribute), `summary()`.
- `positions(time=None)`: `reach_index`, `s`, `status`, `mass` for one
  output time (index or datetime; nearest) as a DataFrame; without `time`
  the full arrays as a Dataset.
- `map_positions(time)`: `(x, y)` from `Network.map_position`, plus the
  polylines for plotting via `polylines()` (a list of `(x, y)` vertex
  arrays per reach).
- `bins(bin_length=100.0)`: a `NetworkBins`; cached per bin length.
- `counts(time, bin_length=100.0)`: particle count per bin.
- `concentration(time, bin_length=100.0, smoothing=None)`: mass per bin
  divided by `width * depth * bin_width` from the hydraulics at that time,
  in `<mass_units> m-3`; NaN where `flow_out == 0`. `smoothing` is a
  kernel bandwidth in meters, or `"auto"` for `sqrt(2 K dt)` of the bin's
  reach; the Gaussian kernel along `s` spreads each particle's mass over
  the bins of its reach only (no kernel across junctions), renormalized so
  mass is conserved per reach. `time` may be a slice or list to get a
  `(time, bin)` DataArray; a `persist(path, bin_length, smoothing)` method
  writes the full cube to a derived NetCDF for animations.
- `reach_concentration(time, ...)`: the same with `bin_length=np.inf`.
- `arrival_times(outlet=None)`: DataFrame of exited particles with
  `exit_time`, `exit_reach`, `reach_id`, `release_*`, `mass`; filtered to
  one outlet id when given. `arrival_histogram(outlet, bin_seconds)` returns
  mass per time bin, which is the breakthrough curve.
- `to_dataframe(time=None)`: long-format DataFrame.
- `to_vtp(output_dir, times=None)`: writes `vtp/network_XXXX.vtp` and a
  `network.pvd` through the existing `VTPWriter` and `PVDWriter`, with
  points from `map_positions` and `reach_index`, `s`, `mass`,
  `source_index` as point data. The existing writers take a particles-like
  object; a small adapter in `results.py` presents the needed attributes.

## Configuration

`NetworkConfig` is a frozen dataclass with validation in `__post_init__`
(types, ranges, `output_interval % dt == 0`, sources non-empty, every
`reach_id` known once a provider is attached, and `times[0] <= start_time
< end_time <= times[-1]` against the provider's time axis). Built by
`NetworkConfig.from_dict(d)` or `NetworkConfig.from_toml(path)`, which
reads the `[network]` table and raises on unknown keys.

| Group | Key | Default | Notes |
|---|---|---|---|
| input | `hydraulics_file` | required | path to the export |
| | `interpolation` | `"linear"` | or `"hold"` |
| | `reach_subset` | `None` | list of ids or `{outlet = id}` |
| | `dtype` | `"float64"` | or `"float32"` |
| time | `start_time` | file first time | ISO datetime |
| | `end_time` | file last time | ISO datetime |
| | `dt` | 900.0 | seconds |
| | `output_interval` | 3600.0 | seconds, integer multiple of `dt` |
| dispersion | `model` | `"fischer"` | `"fischer"`, `"constant"`, `"none"` |
| | `scale` | 1.0 | multiplies the Fischer coefficient |
| | `cap` | `None` | m^2/s |
| | `value` | `None` | m^2/s, required for `"constant"` |
| sources | `particle_mass` | `None` | global particle mass |
| | `mass_units` | `"kg"` | label only |
| | `sources` | required | list of source rows |
| solver | `max_hops` | 1000 | |
| | `seed` | `None` | RNG seed; a `seed` passed to `run_network_simulation` takes precedence |

TOML example:

```toml
[network]
hydraulics_file = "drb_network_hydraulics.nc"
dt = 900.0
output_interval = 3600.0
start_time = "1979-03-01"
end_time = "1979-04-01"
particle_mass = 0.5
mass_units = "kg"

[network.dispersion]
model = "fischer"
scale = 1.0

[[network.sources]]
reach_id = 1234
form = "slug"
time = 0.0
mass = 1000.0

[[network.sources]]
reach_id = 2345
s_frac = 0.5
form = "loading"
rate = 0.01
start = "1979-03-05"
end = "1979-03-10"

[[network.sources]]
reach_id = 3456
form = "concentration"
curve = [["1979-03-02", 0.0], ["1979-03-03", 5.0], ["1979-03-06", 0.0]]
```

Python: `config = NetworkConfig.from_dict({...})` or a plain dict passed
straight to `run_network_simulation`, and `get_network_config_template()`
returning a commented TOML string for notebooks, following
`get_settings_template()`.

## Entry points

```python
def run_network_simulation(
    config: NetworkConfig | dict | str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    seed: int | None = None,
    comm=None,
    quiet: bool = False,
) -> NetworkResults:
```

Orchestration: build the config; open the provider with the config's
subset, dtype, and time window; build the network; expand sources; create
the solver with the RNG from the package's `get_prng(seed)` helper and the
rank's particle slice; print diagnostics; create the writer; loop `step`
and write at output times; write `exit_time` / `exit_reach`; close; return
`NetworkResults(output_dir)` (rank 0 only under MPI; other ranks return
`None`).

Console scripts, mirroring the existing pair:

- `fluvial_particle_network SETTINGS.toml --output DIR [--seed N] [--quiet]`
- `fluvial_particle_network_mpi ...` (same arguments; initializes
  `mpi4py.MPI.COMM_WORLD` and passes it as `comm`).

Package exports added to `fluvial_particle/__init__.py`:
`run_network_simulation`, `NetworkConfig`, `NetworkResults`,
`FileHydraulicsProvider`, `Network`, `estimate_particles`,
`get_network_config_template`.

## Dependencies and packaging

- `pyproject.toml` `dependencies`: add `xarray>=2023.1` and
  `h5netcdf>=1.1` (both pure Python; xarray brings pandas, already in
  dev). `h5py` and `numpy` stay conda-provided as today. `mpi4py` remains
  optional as it is for the existing MPI path.
- `dev` extras: add `matplotlib` (demo notebook) and `scipy` (analytical
  tests: inverse Gaussian, Kolmogorov-Smirnov, normality).
- `environment.yml` unchanged (no new compiled dependencies).
- `.claude/rules/network.md`: a short rules page with the module map, the
  data path locations, and the "read the pywatershed spec first" note.
- Docs: `docs/network.rst` (concepts, convention, sources, concentration,
  MPI), a `[network]` section in `docs/optionsfile.rst`, API entries in
  `docs/api.rst`, a link from `docs/output.rst`.
- Version: bump minor (`0.0.6` to `0.1.0`) when the feature merges; a new
  subpackage and two new dependencies warrant it.

## Testing

All tests use synthetic data built in the test or in
`tests/network/support.py`; the DRB file is not committed. Files:
`tests/network/test_provider.py`, `test_network.py`, `test_sources.py`,
`test_solver.py`, `test_writer_results.py`, `test_analytical.py`,
`test_config_cli.py`.

Support fixtures:

- `three_reach_file(tmp_path, ...)`: writes a schema-conforming NetCDF
  (two headwaters into one outlet, three timestamps, optional polylines,
  optional temperature) using xarray, with chosen hydraulics.
- `chain_file(tmp_path, n_reach, length, velocity, K_target, ...)`: a
  uniform chain with hydraulics solved so the Fischer coefficient equals
  `K_target`.
- `ArrayHydraulicsProvider`: the in-memory protocol implementation.

Provider: every validation error names the variable; `to_index` range,
cycle, and `is_outlet` consistency; `hold` and `linear` values against
hand arithmetic; zero-flow guard on a dry-to-wet interval; the window
holds two slices and advances once per crossed timestamp (spy on the
dataset indexing); backwards request resets with a warning; subset by ids
and by outlet closure remaps `to_index` and truncates the reach dimension;
float32 dtype; out-of-range time raises; chunk warning on a badly chunked
file; polylines are not read until requested.

Network: `headwaters`, `outlets`, `upstream_of`, `parents`; `map_position`
at the ends and midpoint of a three-vertex reach against hand values; NaN
without polylines; `NetworkBins` counts and widths for lengths that divide
evenly and not, `bin_of` at bin edges, `np.inf` gives one bin per reach.

Sources: each form's total mass and count for both resolution modes;
even-quantile release times for a ramp curve; Poisson mode reproducible
with a seed; concentration form uses `flow_in`/`flow_out` interpolation at
`s_frac = 0.5`; `s` vs `s_frac` exclusivity and range; unknown `reach_id`;
outside-window sources; mass conservation to 1e-12 after rounding;
`estimate_particles` reproduces a hand calculation.

Solver (single steps with `ArrayHydraulicsProvider` and `model = "none"`
unless stated): one downstream hop with the time carry giving the exact
landing `s`; two hops; hop into a zero-velocity reach lands at `s = 0`;
exit with the exact `exit_time`; release mid-step advances only `tau`;
dispersive downstream hop with displacement carry; dispersive exit at end
of step; upstream hop into `prev_reach` with the correct `s`; reflect
without history; second upstream overshoot reflects; `max_hops` raises on
an injected cycle; mass conservation (`unreleased + active + exited = N`)
after every step of a multi-step run; identical results for the same seed;
`fischer_coefficient` values, zeros, and cap.

Writer and results: round trip of every variable and attribute on a small
run; `exit_time` written at the end; `positions`, `map_positions`,
`counts` and `concentration` at two bin lengths against hand values
(including NaN on dry reaches); kernel smoothing conserves mass per reach;
`arrival_times` filtering and `arrival_histogram` mass totals;
`to_dataframe`; `to_vtp` writes files that the existing VTP tests can open;
`persist` round trip. MPI: a two-rank test through `mpiexec` on the
writer, skipped unless `mpi4py` imports and `h5py.get_config().mpi` is
true, comparing the file to a serial run with the same seed decomposition.

Config and CLI: `from_toml` on the example above; unknown key raises;
`output_interval` not a multiple of `dt` raises; datetime parsing; the
console script runs the three-reach case end to end and writes the file.

Analytical acceptance tests (`@pytest.mark.slow`, 10-reach uniform chain,
20,000 particles, seeded):

1. `model = "none"`, per-reach velocities varied: every `exit_time` equals
   the sum of `length_i / v_i` to 1e-9 relative, for both `dt = 900` and
   `dt = 86400`.
2. `model = "constant"`, slug at `s = 0` of reach 0: at a fixed `t` before
   any exit, the position mean and variance are within 3 standard errors
   of `v t` and `2 K t`; the total distance along the chain passes a
   normality test (`scipy.stats.normaltest`, p > 0.01).
3. Same, run to completion: exited `exit_time` values pass a
   Kolmogorov-Smirnov test (p > 0.01) against the inverse Gaussian with
   mean `L / v` and shape `L^2 / (2 K)` where `L` is the chain length.

One end-to-end test on the DRB file is skipped unless
`FLUVIAL_PARTICLE_DRB_FILE` points at it: a one-day run from all
headwaters with mass conservation and at least one exit at reach 4205.

## Demo notebook

`notebooks/network-drb-demo.ipynb`, with a first cell holding the path to
the DRB export and a note that it is produced by
`examples/02a_network_hydraulics_export.ipynb` in pywatershed.

1. Open `FileHydraulicsProvider`, build `Network`, print the summary and
   the dt diagnostic; plot the network polylines colored by last-day
   velocity.
2. Sources: an equal-mass slug at every headwater on day one of March 1979,
   plus one continuous loading on a mainstem reach for a week; use
   `estimate_particles` to pick `particle_mass`.
3. Run one month with `dt = 900`, hourly output, seeded.
4. Arrival-time distributions at reach 4205 (Trenton) and the other
   outlets; mass recovered versus released.
5. Concentration along the mainstem (upstream of 4205) as a
   distance-versus-time plot at 500 m bins, raw and smoothed.
6. Map animation of particles on the polylines with matplotlib
   `FuncAnimation`, exported as HTML5 video or GIF, with concentration on
   the mainstem reaches as line color.

## Extension points (not built)

- **In-memory provider and BMI facade** on the solver's initialize /
  update(dt) / finalize; the provider dict keys are the vocabulary.
- **Behavioral particles**: settling as a first-order loss `w_s / depth`
  per step, temperature-dependent rates from `water_temperature`.
- **Distributaries**: a second dimension on `to_index` with per-time flow
  fractions; the downstream hop becomes a flow-weighted random choice.
- **Drift-corrected random walk** across `K` discontinuities (LaBolle et
  al.).
- **Streaming beyond two slices** (prefetch) if I/O ever dominates.
- **Sub-reach hydraulics** if a source ever provides them; bins already
  carry `s_start`/`s_end` for it.
- **Brownian-bridge exit detection**: for a particle ending a step within
  a few sigma of an outlet, exit with the bridge crossing probability
  exp(-2 d1 d2 / sigma^2) and interpolate the crossing time; removes the
  sqrt(dt) first-passage bias.

## References

- Fischer, H.B., List, E.J., Koh, R.C.Y., Imberger, J., Brooks, N.H.
  (1979). Mixing in Inland and Coastal Waters. Academic Press. Chapter 5,
  longitudinal dispersion in rivers, K = 0.011 u^2 W^2 / (d u*).
- LaBolle, E.M., Quastel, J., Fogg, G.E., Gravner, J. (2000). Diffusion
  processes in composite porous media and their numerical integration by
  random walks. Water Resources Research 36(3), 651-662.
- Chhikara, R.S., Folks, J.L. (1989). The Inverse Gaussian Distribution.
  Marcel Dekker. (First-passage time of Brownian motion with drift.)
- pywatershed spec: `docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md`.
