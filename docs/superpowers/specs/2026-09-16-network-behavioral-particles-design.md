# Behavioral particles for the 1D network solver

Date: 2026-09-16
Status: draft for review; amended 2026-09-16 during the PR 1 implementation (Decision 7: mixing
kernel, walk domain, deposition rule, sub-step rule; 7a: reference numbers on the full column;
Decision 8 and the configuration table: `substep_fraction`, `max_substeps` 1000; Testing:
tolerances measured)
Branch: `spec/network-behavioral-particles` (off `main`)
Extends: `2026-09-13-network-particle-solver-design.md` (the "base spec")

## Purpose

Give the network solver the same extensibility the 2D/3D VTK solver has:
a particle base class with documented seams, concrete behaviors as
subclasses, and a registry that lets the options file name the class. The
base spec listed behavioral particles, temperature dependence, and
reactions as non-goals and deferred them to "a later addition to the
solver's per-step update". This addendum replaces that wording with an
extension point, and defines the first three behaviors:

1. **Quasi-2D drift**: a resolved vertical position per particle with a
   log-law velocity profile, parabolic vertical dispersion, settling or
   swimming velocity, and settlement at the bed. This is the 1D form of
   the existing `LarvalTopParticles` / `LarvalBotParticles` /
   `FallingParticles` and the target for larval drift studies.
2. **First-order loss**: decay, mortality, or retention as a per-step
   mass multiplier, optionally temperature dependent.
3. **Carried temperature**: the first constituent that a particle
   transports as a property of the water rather than as released mass,
   with the seeding mode that requires.

It also fixes the architecture for a later **equilibrium chemistry step**
on sub-reach bins without building it.

## Non-goals

- Any change to the 2D/3D solver's classes. One pure numerical function
  (interval reflection) is extracted to a shared module and used by both.
- Equilibrium chemistry, speciation, or a coupling to a chemistry
  library. The operator protocol and the mixing bias are specified; the
  implementation is an extension point.
- Meteorological forcing for temperature. The export has no equilibrium
  temperature; the first temperature model relaxes toward the exported
  `water_temperature`.
- Distributaries, the BMI facade, Brownian-bridge exits, and the LaBolle
  drift correction across longitudinal `K` jumps stay as listed in the
  base spec.

## Facts that shape the design

**The export already carries what the behaviors need.** Required per
step: `velocity`, `depth`, `width`, `ustar`, `flow_in`, `flow_out`.
Required static: `slope`. Optional: `mann_n`, `water_temperature`. So the
log-law profile and the vertical dispersion coefficient need no schema
change, and the concentration post-processing already forms a bin volume
as `width * depth * bin_width`.

**Screening numbers (DRB export, medians: depth 0.59 m, ustar 0.10 m/s,
velocity 0.70 m/s, Fischer K 13 m^2/s).**

- Column mixing time `h^2 / Kz_mean` with `Kz_mean = kappa ustar h / 6`
  is about 90 s. The run step is 900 s. A vertical walk that takes one
  step per `dt` fully mixes the column every step and resolves nothing;
  the vertical model must sub-step (about 150 substeps at these medians
  with the criterion of Decision 7).
- Rouse number `P = w_s / (kappa ustar)` is 0.02 for a 1 mm/s settling
  velocity and 0.24 for 1 cm/s. The vertical model changes results when
  behaviors move at centimetres per second, or in shallow low-shear
  reaches. For a passive tracer at these hydraulics it changes nothing
  measurable, which is the right first analytical test.
- Elder's longitudinal dispersion from the resolved vertical shear,
  `5.93 h ustar`, is 0.35 m^2/s here against a Fischer K of 13. The double
  count when both are on is small for the DRB but not zero, and larger
  in narrow channels; Decision 7a makes it explicit and corrects it by
  default.
- The 2D/3D solver parameterizes every diffusion coefficient as
  `lev + beta_i ustar depth` with user-set `beta` and `lev`; its vertical
  default `beta_z = 0.067` equals the depth mean of the parabolic profile
  at `kappa = 0.41`. The network solver should expose the same choices
  rather than hard-code one profile.

**The current solver is a template method with private seams.** `step`
calls `_release`, `_advect`, `_disperse`. Per-particle velocity is
`v[reach]`. `mass` is a scalar per particle and constant. `status` is
0, 1, 2. `NetworkWriter` has a fixed variable list. `NetworkConfig` has no
particle-type key. `run_network_simulation` constructs `NetworkSolver` by
name. None of these is wrong; together they mean a new behavior is a
library edit.

**What worked and did not in 2D/3D.** The registry-by-name pattern and
small hook overrides (`perturb_z`, `validate_z`, `deactivate_particles`)
worked. Each subclass also overriding `create_hdf5`, `write_hdf5` and
`write_hdf5_xmf` did not: three near-copies of the output code per class.
State and outputs are declared here, not written by subclasses.

**Lessons carried from the base solver:** start from an analytical test
that fails; document a discretization bias with a bound, not silently; a
random walk in a coefficient that varies in space needs the gradient
drift term or it piles particles on the low-coefficient side.

## Decisions

1. **Subclass pattern, mirrored from 2D/3D.** `NetworkSolver` stays the
   base class and the passive model. Behaviors are subclasses in
   `network/particles.py`, registered in `PARTICLE_MODELS` by a short
   name (`"passive"`, `"drift"`, `"decay"`, `"temperature"`). The config
   names the model; the run function looks it up. A user class is named
   as `"package.module:ClassName"` and imported; it must subclass
   `NetworkSolver`.

2. **Two hooks, both no-ops in the base.**
   - `on_release(idx, h)`: called from `_release` with the indices of the
     particles released this step, so a model can initialize its state
     (a drift particle draws its initial relative elevation).
   - `behave(h, tau, t, dt) -> velocity_factor | None`: called after
     release and before advection with the step's hydraulics dict.
     It may change declared state, `mass`, and `status`, and may return a
     per-particle multiplier on the reach velocity for this step's
     advection. `None` means 1.
   `_advect` takes the factor: `s += factor * v[reach] * tau`, and the
   time carry across a hop uses `factor * v` on both sides of the hop.
   Nothing else in `step` changes. The dispersion step, the hop rules,
   and the exit bookkeeping are untouched, so the base spec's analytical
   tests keep their meaning for every model.

3. **Declared state.** A model lists its per-particle arrays as a class
   attribute of `StateVar(name, dtype, shape=(), fill, output=True,
   units="", long_name="", kind="extensive" | "intensive" | None)`, with
   `shape` per particle, `()` or `(k,)`. The base allocates them on
   `__init__` and exposes read-only views; the writer creates an output
   variable with dims `(time, particle)` or `(time, particle, <dim>)` for
   each with `output=True`; `NetworkResults.positions` returns them. A
   `(k,)` state names its dimension and coordinate labels
   (`dim="constituent", labels=("temperature",)`). Models never touch the
   writer.

4. **Mass and constituents.** `mass` remains the particle's weight in
   `mass_units`, set by the source expansion and conserved by release. A
   model may change `mass` in `behave` (first-order loss). Constituents
   are declared state, extensive (a mass carried, summed per bin and
   divided by bin volume) or intensive (a property of the parcel, mass-
   weighted mean per bin). `NetworkResults.concentration(time, ...,
   variable="mass")` gains the `variable` argument and applies the
   declared `kind`. `arrival_histogram` likewise.

5. **Status codes.** `Status` gains `SETTLED = 3` and `REMOVED = 4`.
   Every code above `ACTIVE` is terminal. `exit_time` and `exit_reach`
   become the time and reach at which a particle reached any terminal
   status; the names are kept for the file format, the long_name changes,
   and `arrival_times` filters on `status == EXITED`. `NetworkResults`
   gains `terminal(status)` returning the same frame for any terminal
   code. Mass conservation in tests becomes "sum over all statuses = N".

6. **The solver still never depends on bins or polylines.** Anything
   that needs a bin (equilibrium chemistry) is an operator composed in
   the run loop after `solver.step()`, taking the solver's state views and
   a `NetworkBins`. The solver's write-access surface for such an
   operator is `mass` and the declared state, through a small
   `solver.state` mapping; positions and status are not writable from
   outside. See Decision 12.

7. **Vertical model.** Relative elevation `zeta = z / h` in `[0, 1]` per
   particle, preserved across a reach hop. `zeta_min` (default 0.001, the
   analogue of `vertbound`) is not a wall: the walk lives on the full
   column, `zeta_min` is the clip below which the log-law velocity factor
   is held at its `zeta_min` value (and above `1 - zeta_min` likewise), and
   the thickness of the bed contact layer in which deposition acts. Per
   reach and step, from `ustar`, `depth`, `velocity` and the dispersion
   configuration of Decision 7a:
   - velocity factor `f(zeta)` from the configured velocity profile;
     for the log law `f = 1 + (ustar / (kappa v)) (1 + ln zeta)` evaluated
     at `clip(zeta, zeta_min, 1 - zeta_min)`, floored at 0 so no particle
     moves upstream, and renormalized (closed form) so that
     `integral_0^1 f = 1` exactly: a well-mixed column advects at the
     reach velocity;
   - vertical mixing `Kz(zeta)` from the configured vertical profile;
   - **the mixing kernel leaves a uniform column exactly uniform at any
     sub-step.** For the parabolic profile the Ito walk
     `d zeta = a (1 - 2 zeta) dt + sqrt(2 a zeta (1 - zeta)) dW`,
     `a = kappa ustar / h`, is a Jacobi (Wright-Fisher) diffusion, and with
     `zeta = (1 - cos theta) / 2` it is the polar angle of Brownian motion
     on the unit sphere with generator `a` times the spherical Laplacian
     (Karlin and Taylor 1981). One sub-step is therefore a random rotation:
     a von Mises-Fisher step about the current direction (Fisher 1953;
     sampled by the closed-form inverse CDF of Ulrich 1984 and Wood 1994)
     with the concentration calibrated so the first spherical mode decays
     as the heat kernel does, `E[cos psi] = exp(-2 a dt)`. Isotropy makes
     the uniform measure invariant for every step; higher modes are
     approximated to first order in `a dt`, which is what the sub-step
     rule controls. For the constant and `"value"` profiles the kernel is
     a Gaussian displacement mirror-reflected at 0 and 1 (`reflect_interval`
     from `fluvial_particle/random_walk.py`, shared with `Particles.validate_z`),
     exact for a constant coefficient; a non-zero `background` on the
     parabolic profile adds that step for its part.
     *Measured and rejected:* the Euler-Ito walk with the gradient drift
     `dKz/dz` and a reflecting wall at `zeta_min` (Visser 1997; Ross and
     Sharples 2004), the form this decision first specified. Because `Kz`
     vanishes linearly at the wall, the drift step `a dt` must be far
     smaller than `zeta_min`, about a hundredth of `1 / |Kz''|`, to keep the
     column uniform: at the sub-step rule below it piled particles into
     mid-column (chi-square p ~ 1e-100), biased the mean advection by 1.5
     percent and produced half of Taylor's shear dispersion; fixed-step
     Euler or Visser schemes needed a hundred times more sub-steps and
     per-particle adaptive steps did not converge monotonically.
   - the vertical velocity `w` (positive downward: settling, or negative:
     swimming up, possibly time dependent) is an operator-split
     displacement `-w dt / h` after the mixing kernel, mirror-reflected at
     the surface and the bed;
   - **Deposition at the bed** is a Robin boundary: the flux into the bed
     equals a deposition velocity `k_d` (m/s) times the near-bed
     concentration. `k_d = 0` is a reflecting bed (a dissolved tracer, or
     sediment above its critical shear); `k_d -> inf` is a perfectly
     absorbing bed (attachment on first contact). A particle whose
     settling-shifted position, before reflection, lies in the contact
     layer `[0, zeta_min)` or below it has made contact and deposits with
     the per-contact probability that reproduces `k_d` for that sub-step,
     `p = k_d dt_sub / (zeta_min h + max(w, 0) dt_sub)`, clipped to 1, else
     reflects. The particles making contact per sub-step are those within
     `zeta_min h + w dt_sub` of the bed, so removing each with `p` gives the
     Robin flux while `p < 1`: the layer rule `k_d dt_sub / (zeta_min h)`
     when contact is by mixing, `k_d / w` when settling dominates. The
     probability is derived, never a user parameter, so results do not
     depend on the sub-step count while `p` is small. Two bias bounds are
     stated in the docs: the rule samples the mean density over the contact
     reach rather than the bed density, so the rate is low by about
     `w (zeta_min h + w dt_sub) / (2 K)` relative (3 percent at the default
     sub-step for `w = 1 cm/s`, `K = 0.0067 m^2/s`; it shrinks with the
     sub-step and as `w^2`); and when `(k_d - w) dt_sub` exceeds
     `zeta_min h` the probability clips at 1 and the rate depends on the
     sub-step count (raise `zeta_min` or lower `substep_fraction`; the
     startup diagnostics report the probability, and the solver warns once
     and counts the clipped contacts when it happens). A dry reach (no
     depth) has no contact layer and deposits nothing. When
     `critical_ustar` is set, `k_d` is multiplied by Krone's factor
     `max(0, 1 - (ustar / critical_ustar)^2)`, so nothing deposits where
     the reach shear exceeds the critical value. A deposited particle gets
     `status = SETTLED`, `exit_time = t + dt`, `exit_reach = reach`,
     `zeta = 0`, and its `s` is kept (it is a position on the bed, not NaN).
   - **Sub-stepping.** The walk runs `n_sub = ceil(dt / (c h^2 / Kz_max))`
     substeps with `c = substep_fraction` (default 0.03) and `Kz_max` the
     profile's maximum, capped by `max_substeps` (default 1000); one count
     for all active particles, the maximum over their reaches, so the walk
     stays vectorized; the velocity factor returned to advection is the
     mean of `f(zeta)` over the substeps. Since the kernels keep the column
     well mixed at any step, `c` only sets how well the within-step shear
     dispersion is resolved: 2 percent at 0.03, 12 percent at 0.1. At the
     DRB medians 0.03 is about 520 substeps of vectorized numpy over the
     active particles (about 4 s per step for 100,000 particles); settling
     equilibria need a finer fraction (0.01 for the Rouse profile at
     `P = 0.5`). The startup diagnostics report the count and the reaches
     that hit the cap, and the solver warns once when a step hits it.
   `zeta` is declared state with `output=True`; `NetworkResults.profile
   (time, bin_length, n_zeta=20)` histograms it per bin.

7a. **Dispersion is configurable, following the 2D/3D solver.** There,
   every coefficient is `lev + beta_i * ustar * depth` with `beta` a
   three-vector and `lev` a background value, and the vertical default
   `beta_z = 0.067` is the depth mean of the parabolic profile with
   `kappa = 0.41` (`kappa / 6 = 0.068`). The network exposes the same
   structure in `[network.dispersion]`, longitudinal at the top level as
   today and a `vertical` sub-table read only by models that resolve the
   vertical:

   | Key | Default | Meaning |
   |---|---|---|
   | `model` | `"fischer"` | longitudinal: `"fischer"`, `"constant"`, `"none"` |
   | `scale`, `cap`, `value` | as today | |
   | `background` | 0.0 | m^2/s added to the longitudinal K, the analogue of `lev` |
   | `shear_correction` | `"auto"` | `"auto"`, `"on"`, `"off"`; see below |
   | `vertical.profile` | `"parabolic"` | `"parabolic"`: `Kz = kappa ustar h zeta (1 - zeta)`; `"constant"`: `Kz = beta ustar h`, the 2D/3D form; `"value"`: a fixed `Kz` in m^2/s |
   | `vertical.kappa` | 0.41 | parabolic profile and log law |
   | `vertical.beta` | 0.067 | constant profile |
   | `vertical.value` | none | m^2/s, required for `"value"` |
   | `vertical.background` | 0.0 | m^2/s added to `Kz` |
   | `vertical.scale` | 1.0 | multiplier on `Kz` |
   | `vertical.velocity_profile` | `"log"` | `"log"` or `"uniform"` (`f = 1`) |

   **Shear correction.** Longitudinal dispersion in a river is the sum of
   what vertical shear produces (Elder's term, `5.93 h ustar` for the log
   law with parabolic mixing) and what transverse shear produces, the
   larger part in a wide channel. Fischer's formula is an empirical fit
   to the total and contains both. A model that resolves the vertical
   generates the vertical-shear part from its own kinematics, so the
   kick must supply only the transverse remainder or the vertical part is
   counted twice: 3 percent of Fischer at the DRB medians, 60 percent in
   a 5 m wide, 1 m deep, 0.3 m/s stream. The kick remains a longitudinal
   random walk; the solver has no lateral position, and what the kick
   represents is the longitudinal spreading that transverse shear would
   cause if it were resolved. With `shear_correction = "auto"` the
   correction is applied whenever the model resolves the vertical, the
   velocity profile is not uniform, and a longitudinal model other than
   `"none"` is on (with `"none"` there is nothing to correct), and never
   otherwise; `"on"` and `"off"` override; `"on"` with a model that does
   not resolve the vertical is a config error. The corrected coefficient is
   `max(K - c ustar h, 0)` where `c` is Taylor's dimensionless
   shear-dispersion integral (Taylor 1953) for the configured velocity
   and `Kz` profiles, evaluated by quadrature **on the walk's own
   domain**: the full column, with the velocity factor clipped at
   `zeta_min`, floored at zero and renormalized to unit mean, exactly as
   the walk applies it, and with the cross-sectional average (the
   `1 / width` factor of Taylor's integral). That is what the resolved
   motion generates, and it is what must be removed from the kick; the
   correction is applied to the model's K before `background` is added,
   so the background stays the floor it is documented as. On the full
   column with the unclipped log law the quadrature gives 5.86 (Elder's
   `0.404 / kappa^3` at `kappa = 0.41`); clipping at `zeta_min = 0.001`
   holds the near-bed velocity at its `zeta_min` value and gives 5.83, at
   0.01 it gives 5.58, a 5 percent loss. (The first draft of this
   decision truncated the walk's domain instead, which removed the slow
   near-bed layer altogether: 5.65 at 0.001 and 4.62 at 0.01, a 21
   percent loss, and the reason the floor was set at 0.001; with the clip
   the choice of `zeta_min` is no longer about shear dispersion but about
   the deposition contact layer.) The floor on the velocity factor costs
   another few percent at typical `ustar / v` (5.23 at the DRB median
   ratio 0.143), and once it is active the clip no longer matters. `c`
   depends on the reach only through `ustar / v`, so it is tabulated once
   on a grid of that ratio at solver construction and interpolated per
   reach per step; the per-step cost is nil. The `"value"` profile and a
   non-zero vertical `background` also depend on `ustar h` and are
   evaluated per reach per step by a coarser quadrature. The constant
   `Kz` profile at `beta = 0.067` gives 6.58.

8. **Drift model parameters** (`[network.particles]`, `model = "drift"`):
   `settling_velocity` (m/s, positive down, default 0), `swim_velocity`
   (m/s, positive up, default 0), `diel` (optional table `amplitude`,
   `period`, `phase` making `w` sinusoidal in time, the 1D form of the
   larval `amp`/`period`), `deposition_velocity` (m/s, default 0),
   `critical_ustar` (m/s, optional), `zeta_min`, `substep_fraction`,
   `max_substeps`, `initial_zeta` (`"uniform"` or a number). For a larva, the deposition
   velocity is how readily a competent individual attaches on contact and
   the critical shear is the flow above which it cannot hold. The mixing
   and velocity profiles come from `[network.dispersion.vertical]`, not
   from the model, so every model that resolves the vertical shares them.

9. **First-order loss** (`model = "decay"`): `mass *= exp(-k tau)` per
   step, `k = rate * q10 ** ((T - reference_temperature) / 10)` when
   `water_temperature` is present and `q10` is given, else `rate`. A
   particle whose mass falls below `mass_floor` (default `1e-6` of its
   release mass) becomes `REMOVED` so it stops costing work. The same
   class serves mortality and retention; larval settling uses the drift
   model instead because retention there depends on position.

10. **Carried temperature** (`model = "temperature"`): intensive state
    `temperature` (degC) relaxed toward the reach `water_temperature`,
    `T += (T_reach - T) (1 - exp(-exchange_rate tau))`. Requires
    `water_temperature` in the export; a config error otherwise. Its
    value in the first version is a consistency check on transport (the
    mass-weighted bin mean should reproduce the export where the column
    is well mixed) and the vehicle that exercises intensive constituents
    and water seeding. An equilibrium temperature from meteorology is an
    extension point that needs export additions on the pywatershed side.

11. **Water seeding.** A property of the water needs particles that
    represent all the water. New source form `"water"` with
    `particle_volume` (m^3): at `start_time` particles fill every wet
    reach in proportion to `width * depth * length`, uniformly in `s`;
    for the rest of the run particles are injected at each headwater at
    rate `flow_in / particle_volume` and along each reach at rate
    `(flow_out - flow_in) / particle_volume` when positive (lateral
    inflow), uniformly in `s`. `mass = rho_water * particle_volume` with
    `mass_units` forced to `"kg"` for that source, or `particle_volume`
    itself with `mass_units = "m3"`. Expansion happens at initialization
    like every other source, using the provider's time axis; the count is
    reported by `estimate_particles`. Concentration of an intensive
    variable is then a mass-weighted mean that does not depend on the
    seeding density. The export has no temperature for lateral inflow, so
    a particle injected along a reach takes that reach's
    `water_temperature` at the moment it enters; the docs state this
    assumption. Two diagnostics come with the seeding at no cost and are
    part of the water form: **water age**, an intensive declared state
    `age` (s) that every model advances by `tau` per step, giving
    residence time everywhere in the network as its mass-weighted bin
    mean; and **source fraction**, the share of the water in a bin from
    each headwater or lateral inflow, from the existing `source_index`
    through `NetworkResults.source_fraction(time, bin_length)`.

12. **Reactor operator (specified, not built).** `Reactor` protocol:
    `apply(state: SolverState, bins: NetworkBins, h: Mapping, dt: float)
    -> None`, called by `run_network_simulation` after each `step` when
    the config names one. `SolverState` gives read access to `reach`,
    `s`, `status`, `mass` and write access to `mass` and the declared
    state of active particles. The reference implementation would
    aggregate each extensive constituent per bin, form concentrations
    with the bin volume, call an equilibrium routine per bin, and
    redistribute the result to the bin's particles in proportion to
    their mass. **Mixing bias.** Averaging a constituent over a bin of
    length `Delta` every `dt` is a numerical longitudinal dispersion of
    the constituent of about `Delta^2 / (24 dt)`: 0.5 m^2/s for 100 m bins
    at `dt = 900 s`, 46 m^2/s for 1 km bins, against a median Fischer K
    of 13. The docs must state this bound and the test must measure it.
    Particle-to-particle mass transfer (Benson and Bolster 2016) avoids
    bins and the bias; it is the alternative if the bias ever matters.

## Package layout

```
src/fluvial_particle/
  random_walk.py             reflect_interval(x, lo, hi): the fold from Particles.validate_z,
                             now called by both solvers
  network/
    solver.py                NetworkSolver: base and passive model; hooks; declared-state
                             allocation; velocity factor in _advect; Status with SETTLED, REMOVED
    particles.py             StateVar; PARTICLE_MODELS; DriftParticles, DecayParticles,
                             TemperatureParticles; resolve_model(name_or_path)
    vertical.py              VerticalProfiles (velocity factor, Kz, dKz/dz, Kz_max for the
                             configured profiles), substep_count, vertical_walk,
                             shear_dispersion_coefficient (Taylor integral by quadrature)
    sources.py               form = "water"
    config.py                ParticlesConfig; DispersionConfig gains background,
                             shear_correction and the vertical sub-table; reactor name
    writer.py                declared-state variables and dims; particle_model attrs
    results.py               positions with declared state; concentration(variable=);
                             terminal(status); profile()
    dispersion.py            background term; shear correction applied to the longitudinal K
```

## Configuration

`[network.particles]` is a new table; absent means `model = "passive"`
and the run is bit-identical to today for the same seed.

| Key | Default | Notes |
|---|---|---|
| `model` | `"passive"` | registry name or `"pkg.mod:Class"` |
| drift: `settling_velocity` | 0.0 | m/s, positive down |
| `swim_velocity` | 0.0 | m/s, positive up |
| `diel` | none | `{amplitude, period, phase}` s and m/s |
| `deposition_velocity` | 0.0 | m/s; 0 reflecting bed |
| `critical_ustar` | none | m/s; Krone suppression of deposition above it |
| `zeta_min` | 0.001 | velocity-factor clip and bed contact-layer thickness; see Decision 7 |
| `substep_fraction` | 0.03 | fraction of the column mixing time `h^2 / Kz_max` per sub-step |
| `max_substeps` | 1000 | |
| `initial_zeta` | `"uniform"` | or a number in (zeta_min, 1 - zeta_min) |
| decay: `rate` | required | 1/s |
| `q10`, `reference_temperature` | none | both or neither |
| `mass_floor` | 1e-6 | fraction of release mass |
| temperature: `exchange_rate` | required | 1/s |
| `initial_temperature` | reach value at release | degC |

`[network.dispersion]` gains `background` and `shear_correction`, and the
`[network.dispersion.vertical]` sub-table of Decision 7a. `[network.reactor]`
is reserved: `name` and its parameters, unknown otherwise until built.

Unknown keys for the chosen model raise, as everywhere in the config.

## Output

Declared state variables are written with dims `(time, particle)` or
`(time, particle, <dim>)`, the declared dtype, fill, `units`,
`long_name`, and a `kind` attribute. Global attributes gain
`particle_model` and `particles` (JSON of the table). `status` codes 3
and 4 are documented in the variable's `flag_meanings`.

## Testing

Unit (synthetic files from `tests/network/support.py`):

- Hooks: a test subclass records `on_release` indices and returns a
  factor of 0.5 from `behave`; landing `s` and hop timing halve exactly.
- Declared state round-trips through the writer, including a `(2,)`
  constituent with labels; `positions` returns it; `concentration
  (variable=)` sums an extensive and mass-weights an intensive one
  against hand values.
- Registry: name, dotted path, unknown name, a class that is not a
  subclass, model parameters validated.
- `passive` with `[network.particles]` absent reproduces the base spec's
  solver tests bit for bit with the same seed.
- Vertical functions: `integral f(zeta) dzeta = 1` to 1e-6 by quadrature;
  `Kz` for each profile against hand values, the constant profile's depth
  mean equal to the parabolic one at the defaults;
  `shear_dispersion_coefficient` on the full column returns 5.86 within
  0.5 percent for the log law with the parabolic profile (Elder), 5.83 at
  `zeta_min = 0.001` and 5.58 at 0.01 within 1 percent, 6.58 within 1
  percent for the constant profile, and 0 for the uniform velocity
  profile; the von Mises-Fisher calibration and cosine sampler against
  their closed forms; the sphere step and every profile's `mix` leave a
  uniform column uniform; `shear_correction = "auto"` resolves on and off
  correctly for each model and profile combination and `"on"` raises for
  the passive model; substep count at the DRB medians is about 150 at
  `c = 0.1` and 520 at the default 0.03; `reflect_interval` folds multiple
  crossings and is shared with the 2D/3D tests.
- Status: settled particles keep `s`, get `exit_*`, are excluded from
  advection; `terminal(SETTLED)` returns them; mass conservation over all
  statuses.
- Water seeding: initial count per reach proportional to volume within
  rounding; injection count over a day equals `flow / particle_volume`
  times the day.

Analytical acceptance (`@pytest.mark.slow`, uniform chain, seeded):

1. **Uniform stays uniform.** Passive drift model, `w = 0`, a well-mixed
   column after 1 h of walk: `zeta` histogram against uniform by
   chi-square (p > 0.01). The negative control swaps the kernel for the
   rejected Euler-Ito walk at the same sub-steps and fails (p < 1e-6).
2. **Rouse profile.** `settling_velocity` giving `P = 0.5`, fully
   reflecting bed, after equilibration: `zeta` distribution against the
   Rouse profile truncated to `[zeta_min, 1 - zeta_min]` by
   Kolmogorov-Smirnov distance `D < 0.03` at `substep_fraction = 0.01`
   (measured: 0.049 at 0.03, 0.022 at 0.01, 0.023 at 0.003). The settling
   step is a first-order split, so the criterion is a distance, not a
   p-value (at 20,000 particles p > 0.01 would demand the CDF within 1.2
   points everywhere). `P = 2`, as first specified, and any `P >= 1` are
   out of reach: the profile is not integrable at the bed and a third of
   the truncated reference mass sits below `zeta = 0.01`.
3. **Mean advection preserved.** Well-mixed passive drift, `dispersion =
   none`: mean position after 6 h within 3 standard errors of `v t`.
4. **Shear dispersion emerges.** Same run: position variance grows as
   `2 c ustar h t` within 10 percent for `t >> h^2 / Kz`, with `c` the
   quadrature value for the run's `zeta_min` and `ustar / v`; repeated
   with the constant `Kz` profile; with the shear correction on and
   Fischer on, total variance matches the Fischer value within 10
   percent.
5. **First-order loss.** Decay model: total mass at `t` within 1e-6
   relative of `M exp(-k t)` (exact per particle, so the test is a check
   on bookkeeping across hops and exits).
6. **Temperature consistency.** Water seeding on a chain whose export
   temperature is constant in time and steps between reaches: the
   mass-weighted bin mean converges to the export within 0.1 degC where
   `exchange_rate t >> 1`.
7. **Deposition against the Robin condition.** `w > 0` and a finite
   `deposition_velocity`, uniform release, on the constant mixing profile
   (where the Robin condition is a regular boundary condition; with the
   parabolic profile `Kz` vanishes at the bed and the continuum condition
   degenerates to the settling flux): the deposited fraction at 20 and 30
   min against a finite-volume solution of the 1D advection-diffusion
   equation with the Robin bed condition, within 0.02 at
   `substep_fraction = 0.003`, after the initial transient; repeated with
   the sub-step doubled to show the result moves by less than 0.01; and
   with a very large deposition velocity against the absorbing-bed
   solution within 0.05 (the sqrt(dt) bias of absorbing random walks).
   At the default fraction the bias is the bound of Decision 7 (0.024
   absolute at 30 min here).
8. **Water age.** Water seeding on a chain with `dispersion = none`: the
   mass-weighted mean age at the outlet after the initial fill has
   flushed equals the sum of `length_i / v_i` within one `dt`.

Base spec tests keep passing untouched, which is the proof that Decision
2 left the transport kernel alone.

## Plan

- **PR 1**: Decisions 1 to 8 including 7a, `random_walk.py`, the drift
  model, writer and results generics, config, tests 1 to 4 and 7, docs
  page "Behavioral particles" with the shear-correction and sub-step
  statements.
- **PR 2**: Decisions 9 to 11: decay, temperature, water seeding with age
  and source fraction, `concentration(variable=)`, tests 5, 6 and 8,
  `estimate_particles` for the water form.
- **PR 3 (extension, unscheduled)**: the reactor operator with a
  temperature-only equilibrium as the reference implementation and the
  mixing-bias measurement.

## Extension points (not built)

- Reactor operator and equilibrium chemistry (Decision 12).
- Equilibrium temperature from meteorology (export additions).
- Particle-to-particle mass transfer as a bin-free reactor.
- Behaviors that depend on the network position beyond the reach
  (habitat maps keyed by `reach_id`) through `on_release` and `behave`.
- Everything listed in the base spec.

## References

- Taylor, G.I. (1953). Dispersion of soluble matter in solvent flowing
  slowly through a tube. Proceedings of the Royal Society A 219, 186-203.
  (The shear-dispersion integral.)
- Karlin, S., Taylor, H.M. (1981). A Second Course in Stochastic
  Processes. Academic Press, ch. 15. (The Jacobi / Wright-Fisher diffusion
  and its identity with the polar angle of Brownian motion on the sphere.)
- Fisher, R.A. (1953). Dispersion on a sphere. Proceedings of the Royal
  Society A 217, 295-305. (The von Mises-Fisher distribution.)
- Ulrich, G. (1984). Computer generation of distributions on the m-sphere.
  Applied Statistics 33(2), 158-163; Wood, A.T.A. (1994). Simulation of
  the von Mises Fisher distribution. Communications in Statistics,
  Simulation and Computation 23(1), 157-164. (Sampling the step angle.)
- Visser, A.W. (1997). Using random walk models to simulate the vertical
  distribution of particles in a turbulent water column. Marine Ecology
  Progress Series 158, 275-281. (The Euler-Ito walk with the gradient
  drift; measured and rejected here.)
- Ross, O.N., Sharples, J. (2004). Recipe for 1-D Lagrangian particle
  tracking models in space-varying diffusivity. Limnology and
  Oceanography: Methods 2, 289-302. (Sub-step criteria for that walk.)
- Rouse, H. (1937). Modern conceptions of the mechanics of fluid
  turbulence. Transactions ASCE 102, 463-543. (Suspended-sediment
  profile.)
- Elder, J.W. (1959). The dispersion of marked fluid particles in
  turbulent shear flow. Journal of Fluid Mechanics 5(4), 544-560.
  (K = 5.93 h u*.)
- Fischer et al. (1979), as in the base spec.
- LaBolle et al. (2000), as in the base spec. (Gradient drift term.)
- Benson, D.A., Bolster, D. (2016). Arbitrarily complex chemical
  reactions on particles. Water Resources Research 52(11), 9190-9200.
- Erban, R., Chapman, S.J. (2007). Reactive boundary conditions for
  stochastic simulations of reaction-diffusion processes. Physical
  Biology 4(1), 16-28. (Per-contact probability for a Robin boundary.)
- Krone, R.B. (1962). Flume studies of the transport of sediment in
  estuarial shoaling processes. Hydraulic Engineering Laboratory,
  University of California, Berkeley. (Deposition suppression by bed
  shear.)
- Van Rijn, L.C. (1984). Sediment transport, part II: suspended load
  transport. Journal of Hydraulic Engineering 110(11), 1613-1641.
  (Parabolic-constant vertical mixing profiles.)
