1D River-Network Particle Tracking
==================================

The ``fluvial_particle.network`` subpackage tracks passive, non-reacting particles along a river network
using the *network hydraulics* NetCDF export produced by
``pywatershed.utils.export_network_hydraulics`` (or any file with the same schema). It is separate from
the 2D/3D VTK solver: a network run has its own configuration table, entry points, output file, and
results class.

Input
-----

One NetCDF file with dimensions ``reach``, ``time``, and optionally ``vertex``:

* static per reach: ``reach_id``, ``to_index`` (0-based downstream index, -1 at outlets), ``is_outlet``,
  ``length`` (m), ``slope``; optional polylines ``vertex_x``, ``vertex_y``, ``vertex_dist``,
  ``reach_vertex_start``, ``reach_vertex_count``;
* time-varying ``(time, reach)``: ``flow_in``, ``flow_out`` (m3/s), ``velocity`` (m/s), ``depth``,
  ``width`` (m), ``ustar`` (m/s), optional ``water_temperature``.

Where ``flow_out`` is 0 the hydraulics are 0 and particles wait. The provider validates variables,
units, and topology at open, streams two time slices at a time, and can subset reaches
(``reach_subset = [ids]`` or ``{outlet = id}``).

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
table and exit table on ``particle``, and the run configuration as attributes. ``NetworkResults`` derives:

* ``positions(time)``, ``map_positions(time)``, ``polylines()``;
* ``counts(time, bin_length)``, ``concentration(time, bin_length, smoothing)`` on nearly uniform
  sub-reach bins (mass per ``width * depth * bin width``, in ``mass_units m-3``), ``persist()``;
* ``arrival_times(outlet)`` and ``arrival_histogram(outlet, bin_seconds)`` (breakthrough curves);
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
