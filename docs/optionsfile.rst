=======================
Configuration Files
=======================

Simulation parameters are defined in a configuration file. *fluvial-particle* supports two formats:

* **TOML** (recommended) - Standard configuration format, easy to read and edit
* **Python** (legacy) - Original format, still fully supported

TOML Configuration (Recommended)
=================================

TOML files use a structured, human-readable format. Generate a template with:

.. code-block:: bash

    fluvial_particle --init

This creates ``settings.toml`` in the current directory.

Complete TOML Example
----------------------

.. code-block:: toml

    # Fluvial-particle simulation configuration

    [simulation]
    time = 60.0              # Maximum simulation time [seconds]
    dt = 0.25                # Time step [seconds]
    print_interval = 10.0    # Output interval [seconds]

    [particles]
    type = "Particles"       # Options: Particles, FallingParticles, LarvalParticles,
                             #          LarvalTopParticles, LarvalBotParticles
    count = 100              # Number of particles per processor
    start_location = [5.0, 0.0, 9.5]  # Starting (x, y, z) coordinates
    start_depth_fraction = 0.5        # Optional: initial vertical position (0=bed, 1=surface)

    [particles.physics]
    beta = [0.067, 0.067, 0.067]  # 3D diffusion coefficients
    lev = 0.25                    # Lateral eddy viscosity
    min_depth = 0.02              # Minimum depth threshold [meters]
    vertical_bound = 0.01         # Vertical boundary buffer

    [grid]
    track_3d = true                          # true = 3D velocity field, false = 2D
    file_2d = "./data/mesh_2d.vts"           # Path to 2D mesh file
    file_3d = "./data/mesh_3d.vts"           # Path to 3D mesh file

    [grid.field_map_2d]
    bed_elevation = "Elevation"
    shear_stress = "ShearStress (magnitude)"
    velocity = "Velocity"
    water_surface_elevation = "WaterSurfaceElevation"

    [grid.field_map_3d]
    velocity = "Velocity"

    [output]
    vtp = false              # Also write VTP files for ParaView visualization


TOML Section Reference
-----------------------

[simulation]
^^^^^^^^^^^^

**time** (required): Maximum simulation time in seconds. The simulation ends when all particles leave the domain or this time is reached.

**dt** (required): Time step in seconds.

**print_interval** (required): Output interval in seconds. Data is written to HDF5 files at this interval.


[particles]
^^^^^^^^^^^

**type** (optional): Particle class to use. Options:

* ``"Particles"`` (default) - Standard passive particles
* ``"FallingParticles"`` - Settling particles with fall velocity
* ``"LarvalParticles"`` - Particles with sinusoidal swimming behavior
* ``"LarvalTopParticles"`` - Larval particles biased toward surface
* ``"LarvalBotParticles"`` - Larval particles biased toward bed

**count** (required): Number of particles to simulate *per processor*. In parallel mode with N cores, total particles = N × count.

**start_location** (required): Starting position as ``[x, y, z]`` array, or path to checkpoint/CSV file:

.. code-block:: toml

    # Single point release
    start_location = [5.0, 0.0, 9.5]

    # Resume from checkpoint
    start_location = "./previous_run/particles.h5"

    # Time-varying release from CSV
    start_location = "./release_schedule.csv"

**start_depth_fraction** (optional): Initial vertical position as fraction of water depth (0=bed, 1=surface). Default: None (uses z from start_location).


[particles.physics]
^^^^^^^^^^^^^^^^^^^

**beta** (optional): Diffusion coefficients as ``[x, y, z]`` or scalar. Default: ``[0.067, 0.067, 0.067]``

**lev** (optional): Lateral eddy viscosity. Default: 0.25

**min_depth** (optional): Minimum water depth threshold in meters. Default: 0.02

**vertical_bound** (optional): Vertical boundary buffer as fraction of depth. Default: 0.01


[particles.falling]
^^^^^^^^^^^^^^^^^^^

For ``type = "FallingParticles"`` only:

**radius** (optional): Particle radius in meters. Default: 0.0005

**density** (optional): Particle density in kg/m³. Default: 2650.0

**c1** (optional): Viscous drag coefficient. Default: 20.0

**c2** (optional): Turbulent wake drag coefficient. Default: 1.1


[particles.larval]
^^^^^^^^^^^^^^^^^^

For ``LarvalParticles``, ``LarvalTopParticles``, or ``LarvalBotParticles``:

**amplitude** (optional): Swimming amplitude as fraction of depth. Default: 0.2

**period** (optional): Swimming period in seconds. Default: 60.0


[grid]
^^^^^^

**track_3d** (required): Use 3D velocity field (``true``) or 2D (``false``).

**file_2d** (required): Path to 2D mesh file. Supported formats:

* ``.vts`` - VTK XML Structured Grid (**recommended**)
* ``.vtk`` - VTK legacy format
* ``.npz`` - NumPy compressed archive

**file_3d** (required): Path to 3D mesh file. Required even for 2D simulations (use any valid file).


[grid.field_map_2d]
^^^^^^^^^^^^^^^^^^^

Maps standard field names to your model's naming convention.

**Required fields:**

* ``bed_elevation``: Bed/bottom elevation
* ``velocity``: Velocity vector
* ``water_surface_elevation``: Water surface elevation

**Shear velocity source** (at least one required):

* ``shear_stress``: Bed shear stress [Pa] (most common)
* ``ustar``: Direct shear velocity [m/s]
* ``manning_n``: Manning's n field
* ``chezy_c``: Chézy C field
* ``darcy_f``: Darcy-Weisbach f field
* ``energy_slope``: Energy slope field
* ``tke``: Turbulent kinetic energy [m²/s²]

**Optional:**

* ``wet_dry``: Wet/dry indicator (1=wet, 0=dry). Auto-computed from depth if omitted.


[grid.field_map_3d]
^^^^^^^^^^^^^^^^^^^

**Required:**

* ``velocity``: 3D velocity vector


[grid.friction]
^^^^^^^^^^^^^^^

Scalar friction coefficients (alternative to field mapping):

.. code-block:: toml

    [grid.friction]
    manning_n = 0.03      # Manning's n
    # chezy_c = 50.0      # OR Chézy C
    # darcy_f = 0.02      # OR Darcy-Weisbach f
    water_density = 1000.0  # Water density [kg/m³], default 1000


[grid.time_varying]
^^^^^^^^^^^^^^^^^^^

For unsteady flow simulations with time-varying velocity fields:

.. code-block:: toml

    [grid.time_varying]
    enabled = true
    file_pattern_2d = "./data/flow_2d_{}.vts"  # {} = file index
    file_pattern_3d = "./data/flow_3d_{}.vts"
    start_index = 0        # First file index
    end_index = 10         # Last file index (inclusive)
    dt = 60.0              # Time between grid files [seconds]
    start_time = 0.0       # Simulation time of first grid file
    interpolation = "linear"  # linear, nearest, or hold


[output]
^^^^^^^^

**vtp** (optional): Write VTP files for ParaView in addition to HDF5. Default: false


Programmatic Configuration (Notebooks)
=======================================

For Jupyter notebooks and scripts, use the programmatic API instead of files:

.. code-block:: python

    from fluvial_particle import get_default_config, run_simulation, save_config

    # Get default configuration as a nested dictionary
    config = get_default_config()

    # Modify settings programmatically
    config["particles"]["count"] = 200
    config["particles"]["start_location"] = [5.0, 0.0, 9.5]
    config["simulation"]["time"] = 120.0
    config["grid"]["file_2d"] = "./data/mesh_2d.vts"
    config["grid"]["file_3d"] = "./data/mesh_3d.vts"

    # Option 1: Run directly with dict (no file needed)
    results = run_simulation(config, output_dir="./output")

    # Option 2: Save to file first
    save_config(config, "my_settings.toml")
    results = run_simulation("my_settings.toml", output_dir="./output")

    # Access results
    print(f"Simulated {results.num_particles} particles over {results.num_timesteps} timesteps")
    positions = results.get_positions(timestep=-1)  # Final positions
    df = results.to_dataframe()  # Full particle trajectories as DataFrame


Python Configuration (Legacy)
==============================

Python configuration files are still fully supported for backwards compatibility.

To generate a Python template:

.. code-block:: bash

    fluvial_particle --init --format python

Example Python configuration:

.. code-block:: python

    """Options file for fluvial particle model."""
    from fluvial_particle.Particles import Particles

    # Required parameters
    file_name_2d = "./data/mesh_2d.vts"
    file_name_3d = "./data/mesh_3d.vts"
    SimTime = 60.0
    dt = 0.25
    PrintAtTick = 10.0
    Track3D = 1
    NumPart = 100
    StartLoc = (5.0, 0.0, 9.5)
    ParticleType = Particles

    # Field mappings
    field_map_2d = {
        "bed_elevation": "Elevation",
        "shear_stress": "ShearStress (magnitude)",
        "velocity": "Velocity",
        "water_surface_elevation": "WaterSurfaceElevation",
    }
    field_map_3d = {"velocity": "Velocity"}

    # Optional parameters
    beta = (0.067, 0.067, 0.067)
    lev = 0.25
    min_depth = 0.02
    vertbound = 0.01


TOML to Python Key Mapping
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 40

   * - TOML Key
     - Python Key
   * - ``simulation.time``
     - ``SimTime``
   * - ``simulation.dt``
     - ``dt``
   * - ``simulation.print_interval``
     - ``PrintAtTick``
   * - ``particles.type``
     - ``ParticleType``
   * - ``particles.count``
     - ``NumPart``
   * - ``particles.start_location``
     - ``StartLoc``
   * - ``particles.start_depth_fraction``
     - ``startfrac``
   * - ``particles.physics.beta``
     - ``beta``
   * - ``particles.physics.lev``
     - ``lev``
   * - ``particles.physics.min_depth``
     - ``min_depth``
   * - ``particles.physics.vertical_bound``
     - ``vertbound``
   * - ``particles.falling.radius``
     - ``radius``
   * - ``particles.falling.density``
     - ``rho``
   * - ``particles.larval.amplitude``
     - ``amp``
   * - ``particles.larval.period``
     - ``period``
   * - ``grid.track_3d``
     - ``Track3D``
   * - ``grid.file_2d``
     - ``file_name_2d``
   * - ``grid.file_3d``
     - ``file_name_3d``
   * - ``output.vtp``
     - ``output_vtp``


Shear Velocity (u*) Configuration
==================================

*fluvial-particle* uses shear velocity (u*) to compute turbulent diffusion. Multiple methods are supported, listed in priority order:

1. **Direct u* field** - Map ``ustar`` in field_map_2d
2. **Bed shear stress** - Map ``shear_stress`` (most common)
3. **Manning's n** - Field or scalar ``manning_n``
4. **Chézy C** - Field or scalar ``chezy_c``
5. **Darcy-Weisbach f** - Field or scalar ``darcy_f``
6. **Energy slope** - Map ``energy_slope``
7. **TKE** - Map ``tke``

To force a specific method, use ``ustar_method``:

.. code-block:: toml

    [grid.friction]
    manning_n = 0.03
    ustar_method = "manning"  # Force this method even if others available
