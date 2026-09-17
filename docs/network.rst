1D River-Network Particle Tracking
==================================

The ``fluvial_particle.network`` subpackage tracks particles along a river network using the *network
hydraulics* NetCDF export produced by ``pywatershed.utils.export_network_hydraulics`` (or any file with
the same schema). The default particle is passive and non-reacting; behavioral models (a resolved
vertical position with settling, swimming and deposition) are selected by name, see
`Behavioral particles`_. It is separate from the 2D/3D VTK solver: a network run has its own
configuration table, entry points, output file, and results class.

Input
-----

One NetCDF file with dimensions ``reach``, ``time``, and optionally ``vertex``:

* static per reach: ``reach_id``, ``to_index`` (0-based downstream index, -1 at outlets), ``is_outlet``,
  ``length`` (m), ``slope``; optional polylines ``vertex_x``, ``vertex_y``, ``vertex_dist``,
  ``reach_vertex_start``, ``reach_vertex_count``;
* time-varying ``(time, reach)``: ``flow_in``, ``flow_out`` (m3/s), ``velocity`` (m/s), ``depth``,
  ``width`` (m), ``ustar`` (m/s), optional ``water_temperature``.

Where ``flow_out`` is 0 the file's hydraulics are 0 and particles wait. Under
``interpolation = "linear"`` velocity and ustar are additionally forced to 0 whenever either
bracketing day is dry, so a particle never advects on a half-interpolated velocity into a reach that
has no water; flow itself still interpolates. Under ``interpolation = "hold"`` no masking is applied
and the file's own zeros are what the solver sees. The provider validates variables, units, field
values (finite and non-negative), and topology at open, streams two time slices at a time, and can
subset reaches (``reach_subset = [ids]`` or ``{outlet = id}``).

Particle convention
-------------------

A particle's state is ``(reach index, s)`` with ``0 <= s <= length`` from the reach's upstream end.
Each step: exact advection ``s += velocity * tau`` with hops to ``to_index`` carrying the unused time
(so any dt gives the same advective path), then one dispersive kick ``N(0, 1) * sqrt(2 K tau)`` with
displacement carry across boundaries, where ``tau`` is the particle's time in the step: ``dt``, or
less for a particle released mid-step. Upstream overshoot returns to the reach the particle came
from; with no such history it enters one of the reach's parents, picked in proportion to that step's
``flow_out`` of the parents (so a particle may enter a tributary it never visited), and it reflects
only at a true headwater. At an outlet the particle exits and its exit time is recorded.

``K`` is the Fischer coefficient ``0.011 v^2 w^2 / (d u*)`` times ``dispersion.scale``, optionally
capped; ``model = "constant"`` and ``"none"`` are available.

Sources
-------

Every source is a mass loading at ``(reach_id, s)``:

* ``form = "slug"``: ``time``, ``mass``;
* ``form = "loading"``: constant ``rate`` on ``[start, end)``, or ``curve = [[time, rate], ...]``;
* ``form = "concentration"``: ``value`` on ``[start, end)`` or ``curve``; multiplied by the flow at the
  release point (interpolated between ``flow_in`` and ``flow_out``).

Particles sample the loading at equal-mass quantiles (or Poisson with ``spacing = "poisson"``). Set a
global ``particle_mass`` or a per-source ``particles`` count; ``estimate_particles`` reports the budget
for a target number of particles per bin.

Behavioral particles
--------------------

``[network.particles]`` names the particle model; absent or ``model = "passive"``, the run is
bit-identical, for the same seed, to a run made before the table existed.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``model``
     - Behavior
   * - ``"passive"``
     - The transport kernel alone: advection with time carry and one dispersive kick per step.
   * - ``"drift"``
     - Quasi-2D drift: a resolved relative elevation per particle with a log-law velocity profile,
       vertical mixing, settling or swimming, and deposition at the bed (below).
   * - ``"package.module:ClassName"``
     - A user class subclassing ``NetworkSolver`` and overriding the ``on_release`` and ``behave``
       hooks; it declares its per-particle state in ``STATE`` as ``StateVar`` entries, which the
       writer and ``NetworkResults`` handle generically.

**Drift model.** Each particle carries ``zeta = z / h`` (0 at the bed, 1 at the surface), drawn
uniformly at release (or set by ``initial_zeta``) and preserved across reach hops. The particle
advects at ``u(zeta) / v`` times the reach velocity with the log law
``u / v = 1 + (u* / (kappa v)) (1 + ln zeta)``, evaluated at ``clip(zeta, zeta_min, 1 - zeta_min)``,
floored at zero (no particle moves upstream) and normalized so a well-mixed column advects at the
reach velocity exactly. Per step the vertical walk runs ``n_sub`` sub-steps, each an operator split of:

1. mixing by a kernel that leaves a uniform column *exactly* uniform at any step: for the parabolic
   profile the walk is the polar angle of Brownian motion on a sphere (the Jacobi diffusion identity,
   Karlin and Taylor 1981), so one sub-step is a random rotation, a von Mises-Fisher step (Fisher
   1953) calibrated so the first spherical mode decays as the heat kernel does; for a constant
   profile it is a Gaussian displacement mirror-reflected at the bed and the surface;
2. the vertical velocity ``w`` (positive down: ``settling_velocity - swim_velocity`` plus the optional
   ``diel`` sinusoid ``amplitude * sin(2 pi t / period + phase)``) as a displacement reflected at the
   surface and the bed;
3. deposition: a particle whose shifted position lies in the bed contact layer ``zeta < zeta_min`` (or
   below the bed) has made contact and deposits with the derived probability below, else reflects.

The sub-step count is ``ceil(dt / (substep_fraction * h^2 / Kz_max))`` per reach, one count for all
active particles (the maximum, capped by ``max_substeps``). Because the kernels keep the column well
mixed at any step, the fraction only sets how well the shear dispersion within a step is resolved:
about 2 percent at the default 0.03, 12 percent at 0.1. At the Delaware River Basin medians (depth
0.59 m, ``u*`` 0.10 m/s) the default is about 520 sub-steps per 900 s step, roughly 4 s of vectorized
numpy per step for 100,000 particles; settling equilibria near the bed need a finer fraction (0.01
reproduces the Rouse profile at Rouse number 0.5 to a Kolmogorov-Smirnov distance of 0.02). The
startup report prints the count and the reaches that hit the cap, and the solver warns once when a
step hits it (one shallow, high-shear reach holding particles throttles every particle that step).

**Deposition** is a Robin boundary condition: the flux into the bed is a deposition velocity
``deposition_velocity`` (m/s) times the near-bed concentration. Zero is a reflecting bed (a dissolved
tracer, or sediment above its critical shear); a very large value is a perfectly absorbing bed
(attachment on first contact). For a larva the deposition velocity is how readily a competent
individual attaches on contact, and ``critical_ustar`` is the flow above which it cannot hold: the
deposition velocity is multiplied by Krone's factor ``max(0, 1 - (u* / critical_ustar)^2)`` (Krone
1962), so nothing deposits where the reach shear exceeds it. The per-contact probability is derived,
never a parameter: the particles making contact in a sub-step are those within
``zeta_min h + w dt_sub`` of the bed, so removing each with
``p = k_d dt_sub / (zeta_min h + w dt_sub)`` gives the Robin flux (the layer rule when contact is by
mixing, ``k_d / w`` when settling dominates). A deposited particle gets status 3 (``settled``), keeps
its position on the bed (``s`` is kept, ``zeta`` is 0), and ``exit_time`` and ``exit_reach`` record
where and when; ``NetworkResults.terminal(3)`` lists them.

.. list-table::
   :header-rows: 1
   :widths: 24 12 64

   * - Key
     - Default
     - Meaning
   * - ``settling_velocity``
     - 0.0
     - m/s, positive down
   * - ``swim_velocity``
     - 0.0
     - m/s, positive up
   * - ``diel``
     - none
     - table ``{amplitude, period, phase}`` (m/s, s, rad): a sinusoidal vertical velocity
   * - ``deposition_velocity``
     - 0.0
     - m/s; 0 is a reflecting bed
   * - ``critical_ustar``
     - none
     - m/s; Krone's factor scales the deposition velocity by ``1 - (u* / critical_ustar)^2``, zero above it
   * - ``zeta_min``
     - 0.001
     - velocity-factor clip and bed contact-layer thickness, as a fraction of the depth
   * - ``substep_fraction``
     - 0.03
     - fraction of the column mixing time ``h^2 / Kz_max`` per sub-step
   * - ``max_substeps``
     - 1000
     - cap on the sub-step count
   * - ``initial_zeta``
     - ``"uniform"``
     - or a number strictly inside ``(zeta_min, 1 - zeta_min)``

Dispersion configuration
------------------------

``[network.dispersion]`` holds the longitudinal settings and, for models that resolve the vertical,
a ``vertical`` sub-table. Both follow the 2D/3D solver's parameterization ``lev + beta * u* * depth``:
``background`` is the analogue of ``lev`` and ``vertical.beta`` of ``beta``; the parabolic profile's
depth mean at ``kappa = 0.41`` (``kappa / 6 = 0.068``) is the 2D/3D vertical default 0.067.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Key
     - Default
     - Meaning
   * - ``model``
     - ``"fischer"``
     - longitudinal: ``"fischer"``, ``"constant"``, ``"none"``
   * - ``scale``, ``cap``, ``value``
     - 1.0, none, none
     - Fischer multiplier and cap; K for ``"constant"``
   * - ``background``
     - 0.0
     - m2/s added to the longitudinal K on every wet reach, for every model
   * - ``shear_correction``
     - ``"auto"``
     - ``"auto"``, ``"on"``, ``"off"``; see below
   * - ``vertical.profile``
     - ``"parabolic"``
     - ``"parabolic"``: ``Kz = kappa u* h zeta (1 - zeta)`` (van Rijn 1984); ``"constant"``:
       ``Kz = beta u* h``, the 2D/3D form; ``"value"``: a fixed ``Kz`` in m2/s
   * - ``vertical.kappa``
     - 0.41
     - parabolic profile and log law
   * - ``vertical.beta``
     - 0.067
     - constant profile
   * - ``vertical.value``
     - none
     - m2/s, required for ``"value"``
   * - ``vertical.background``
     - 0.0
     - m2/s added to ``Kz``
   * - ``vertical.scale``
     - 1.0
     - multiplier on ``Kz``
   * - ``vertical.velocity_profile``
     - ``"log"``
     - ``"log"`` or ``"uniform"`` (every particle at the reach velocity)

**Shear dispersion and the Fischer coefficient.** Longitudinal dispersion in a river is the sum of
what vertical shear produces (Elder's term, ``5.86 h u*`` for the log law with parabolic mixing at
``kappa = 0.41``, 5.93 in Fischer et al.'s tabulation; Elder 1959) and what transverse shear
produces, the larger part in a wide channel; Fischer's formula (Fischer et al. 1979) is an empirical
fit to the total and contains both. A model that resolves the vertical generates the vertical-shear
part from its own kinematics, so the kick must supply only the transverse remainder or the vertical
part is counted twice: 3 percent of Fischer at the Delaware medians, 60 percent in a 5 m wide, 1 m
deep, 0.3 m/s stream with ``u*`` 0.05 m/s. With ``shear_correction = "auto"`` the
correction is applied whenever the model resolves the vertical, the velocity profile is not
uniform, and a longitudinal model other than ``"none"`` is on (with ``"none"`` there is nothing to
correct and the resolved shear is the run's only longitudinal dispersion); ``"on"`` and ``"off"``
override, and ``"on"`` with a model that does not resolve the vertical is a configuration error. The corrected coefficient is
``max(K - c u* h, 0)`` with ``c`` Taylor's dimensionless shear-dispersion integral (Taylor 1953) for
the configured velocity and mixing profiles, computed by quadrature on the walk's own domain: the
full column, with the velocity factor clipped at ``zeta_min``, floored and renormalized exactly as
the walk applies it; the correction is applied to the model's K before ``background`` is added, so
the background stays a floor. Where the resolved shear exceeds the model's K the reach gets K = 0
and the solver warns once (the vertical walk then generates more shear dispersion than the
longitudinal model claims exists). On the full column with the unclipped log law ``c`` is Elder's
``0.404 / kappa^3 = 5.86``; the clip at 0.001 gives 5.83 and at 0.01 gives 5.58; the constant profile
at ``beta = 0.067`` gives 6.58; the floor costs a few percent at typical ``u* / v`` (5.23 at the
Delaware median ratio 0.143). For the parabolic and constant profiles without a vertical background
``c`` depends on the reach only through ``u* / v`` and is tabulated once at start-up; the ``"value"``
profile and a non-zero ``vertical.background`` also depend on ``u* h`` and are evaluated per reach on
every step by a coarser quadrature, a visible cost on a large network. The acceptance tests show the
resolved shear reproduces ``2 c u* h t`` within 10 percent and that, with Fischer on and the
correction active, the total variance is ``2 K t``.

Running
-------

.. code-block:: python

    from fluvial_particle import run_network_simulation, get_network_config_template

    results = run_network_simulation("network.toml", "./out", seed=42)
    print(results.summary())

or ``fluvial_particle_network network.toml -o out`` (``fluvial_particle_network_mpi`` under
``mpiexec``). ``--init`` prints the settings template.

Output and post-processing
--------------------------

``network_particles.nc`` holds ``reach_index``, ``s``, ``status`` on ``(time, particle)``, the release
table and exit table on ``particle``, and the run configuration as attributes (``particle_model`` and
the ``particles`` table among them). ``status`` is 0 unreleased, 1 active, 2 exited, 3 settled, 4
removed; every code above 1 is terminal, and ``exit_time`` and ``exit_reach`` record when and where a
particle reached it. A model's declared state (``zeta`` for the drift model) is written on
``(time, particle)`` with its units and long name, marked by the attribute
``fluvial_particle_state``. ``NetworkResults`` derives:

* ``positions(time)``, ``map_positions(time)``, ``polylines()``;
* ``counts(time, bin_length)``, ``concentration(time, bin_length, smoothing)`` on nearly uniform
  sub-reach bins (mass per ``width * depth * bin width``, in ``mass_units m-3``), ``persist()``;
* ``arrival_times(outlet)`` and ``arrival_histogram(outlet, bin_seconds)`` (breakthrough curves);
  ``terminal(status)`` gives the same frame for settled (3) or removed (4) particles;
* ``state_variables`` and the declared state as extra columns of ``positions(time)``;
  ``profile(time, bin_length, n_zeta)`` histograms the drift model's ``zeta`` per bin;
* ``to_dataframe()``, ``to_vtp()`` for ParaView.

MPI
---

Particles are split into contiguous slices across ranks; every rank holds the hydraulics. The writer
uses h5py's ``mpio`` driver, which requires an MPI-enabled h5py build.

Approximations
--------------

``K`` is sampled from the reach where the particle ends its advective move; a plain random walk across
a jump in ``K`` slightly over-populates the low-``K`` side; an upstream hop with no recorded history
picks a parent by flow share (a Fickian approximation to upstream spreading: the particle may enter a
tributary it never visited); dispersive exits are stamped at the end of the step. First-passage times are therefore late by at most about ``0.5826 * sqrt(2 K dt) / v + dt / 2``
(the Broadie-Glasserman-Kou continuity correction plus end-of-step stamping); this bound is tight when
the dispersive kick dominates the step's displacement (``sqrt(2 K dt) >> v * dt``), and the bias is much
smaller when advection dominates, because most exits then occur in the exactly monitored advective
substep. The exit-time bias scales as ``sqrt(dt)``, not ``dt``.

For the drift model: a deposited particle keeps the ``s`` it had at the start of the step and is
stamped ``exit_time = t + dt`` (a first-order timing approximation). The per-contact deposition
probability samples the mean concentration over the contact reach ``zeta_min h + w dt_sub`` rather
than the bed concentration, so the deposition rate is low by about
``w (zeta_min h + w dt_sub) / (2 Kz)`` relative: 3 percent at the default sub-step for a 1 cm/s
settling velocity in a 0.0067 m2/s column, shrinking with the sub-step and as ``w^2``. When
``(k_d - w) dt_sub`` exceeds ``zeta_min h`` the probability clips at 1 and the rate depends on the
sub-step count; the startup report prints the probability for the first hydraulics slice, and the
solver warns once and counts the clipped contacts (``clipped_contacts``) when it happens in a run
(raise ``zeta_min`` or lower ``substep_fraction``). A dry reach has no contact layer and deposits
nothing. A perfectly absorbing bed carries the ``sqrt(dt)`` bias of absorbing random walks. Rouse
numbers ``w / (kappa u*)`` of 1 and above concentrate particles in a bed layer thinner than any
affordable sub-step resolves; the model then reproduces the near-bed profile only qualitatively.
When the sub-step count hits ``max_substeps`` the vertical walk resolves less of the shear
dispersion than the fraction implies. The first design used the Euler-Ito walk with the gradient
drift term and a reflecting wall at ``zeta_min`` (Visser 1997; Ross and Sharples 2004); with a
mixing coefficient that vanishes at the wall it could not keep a column uniform at any affordable
sub-step, which is why the kernels above were built instead.

References
----------

* Elder, J.W. (1959). The dispersion of marked fluid particles in turbulent shear flow. *Journal of
  Fluid Mechanics* 5(4), 544-560.
* Fischer, H.B., List, E.J., Koh, R.C.Y., Imberger, J., Brooks, N.H. (1979). *Mixing in Inland and
  Coastal Waters*. Academic Press.
* Fisher, R.A. (1953). Dispersion on a sphere. *Proceedings of the Royal Society A* 217, 295-305.
* Karlin, S., Taylor, H.M. (1981). *A Second Course in Stochastic Processes*. Academic Press, ch. 15.
* Krone, R.B. (1962). *Flume studies of the transport of sediment in estuarial shoaling processes*.
  Hydraulic Engineering Laboratory, University of California, Berkeley.
* Ross, O.N., Sharples, J. (2004). Recipe for 1-D Lagrangian particle tracking models in
  space-varying diffusivity. *Limnology and Oceanography: Methods* 2, 289-302.
* Rouse, H. (1937). Modern conceptions of the mechanics of fluid turbulence. *Transactions ASCE*
  102, 463-543.
* Taylor, G.I. (1953). Dispersion of soluble matter in solvent flowing slowly through a tube.
  *Proceedings of the Royal Society A* 219, 186-203.
* Ulrich, G. (1984). Computer generation of distributions on the m-sphere. *Applied Statistics*
  33(2), 158-163; Wood, A.T.A. (1994). Simulation of the von Mises Fisher distribution.
  *Communications in Statistics, Simulation and Computation* 23(1), 157-164.
* van Rijn, L.C. (1984). Sediment transport, part II: suspended load transport. *Journal of
  Hydraulic Engineering* 110(11), 1613-1641.
* Visser, A.W. (1997). Using random walk models to simulate the vertical distribution of particles
  in a turbulent water column. *Marine Ecology Progress Series* 158, 275-281.
