=======================
The user options file
=======================

Simulation parameters are set in a user options file that is read in by *fluvial-particle* and evaluated as a Python script. See the examples section for a sample user options file.

Some parameters are required in the options file, while others are optional.

Required keyword arguments
============================

**file_name_2d**, str: Path to the 2D input mesh.

**file_name_3d**, str: Path to the 3D input mesh. In the case of a 2D simulation, this entry will be ignored but it is still required.

**SimTime**, float: Maximum allowable simulation time, in seconds. The simulation will end before this time only if all of the tracked particles leave the domain before SimTime.

**dt**, float: The time step used in the simulation, in seconds.

**PrintAtTick**, float: The printing interval, in seconds. Data will be printed to stdout and written to the HDF5 files at intervals of less than or equal to PrintAtTick.

**Track3D**, bool (int 0 or 1): Indicates whether to run a 2D or 3D simulation. If set to 1, the simulation will be 3D and velocity vectors will be interpolated from the 3D mesh. Else, the simulation will be 2D with velocities from the 2D mesh.

**NumPart**, int: The number of particles to simulate, *per core*. This means that in parallel execution mode with *N* cores, the total number of simulated particles will be *N* * NumPart.

**StartLoc**, tuple or str: Indicates the starting location of the particles, can be given as a tuple of length 3 or as a string. Particles can be initialized from a single point, from an existing HDF5 file of particle locations, or from a time-delayed collection of points specified in a CSV file.

* tuple (x, y, z): all particles will be iniated from the same point
* str: path to a file from which thes starting positions will be loaded; either an HDF5 or CSV file

 - ".h5" suffix: the starting locations will be loaded from the HDF5 file as a checkpoint file generated from a previous *fluvial-particle* simulation. The checkpoint file must have the same *total* number of particles as the current simulation, i.e. summed across all CPU cores. By default, data are loaded from the final entry in the HDF5 file, but a particular index to slice into along the time dimension (axis 0) can be provided with the *StartIdx* optional keyword argument.
 - ".csv" suffix: the starting locations will be loaded from a CSV file with 5 columns: start_time, x, y, z, numpart. For example, if a given row in the CSV file is "10.0, 6.14, 9.09, 10.3, 100", then 100 particles will be iniated from the point (6.14, 9.09, 10.3) starting at a simulation time of 10.0 seconds.

**ParticleType**: The type of particles to simulate, either the Particles class or a subclass (e.g. LarvalBotParticles). This argument should not be placed inside quotes or brackets of any kind.


Optional keyword arguments
============================

These arguments can be specified in the options file. Otherwise, the default values will be used.

**beta**, float or tuple: Scales the 3D diffusion coefficients, can be specified as a scalar or a tuple of length 3. Default value: (0.067, 0.067, 0.067)

**lev**, float: Lateral eddy viscosity. Default value: 0.25

**min_depth**, float: The minimum depth that a particle may enter. If a depth update is less than min_depth, then the update is not permitted. Default value: 0.02

**StartIdx**, int: The index used to slice into the HDF5 particles checkpoint file along the 0th axis, i.e. the printing step axis. Only used if an HDF5 file is provided via the StartLoc keyword argument. Default value: -1

**startfrac**, float: If provided, will initialize particles to a vertical position as bed elevation + water depth multiplied with startfrac. startfrac should be between 0 and 1 -- values outside this range will initialize particles at the bed and water surface, respectively. A numpy array of length NumPart can also be used to vary the startfrac for every particle. Default value: None

**vertbound**, float: Bounds the particles in the fractional water column of [vertbound, 1-vertbound]. This prevents particles from moving out of the vertical domain, either by going below the channel bed or above the water surface. Default value: 0.01


LarvalParticles optional keyword arguments
=============================================

**amp**, float or np.ndarray: The amplitude of larval sinusoidal swimming behavior. If a NumPy ndarray is provided, then it must be 1D and have length equal to NumPart. Only for the LarvalParticles subclasses. Default value: 0.2

**period**, float or np.ndarray: The temporal period of larval sinusoidal swimming behavior, in seconds. If a NumPy ndarray is provided, then it must be 1D and have length equal to NumPart. Only for the LarvalParticles subclasses. Default value: 60.0


FallingParticles optional keyword arguments
=============================================

**c1**, float or np.ndarray: The viscous drag coefficient, dimensionless. If a NumPy ndarray is provided, then it must be 1D and have length equal to NumPart. Only for the FallingParticles subclasses. Default value: 20.0

**c2**, float or np.ndarray: The turbulent wake drag coefficient, dimensionless. If a NumPy ndarray is provided, then it must be 1D and have length equal to NumPart. Only for the FallingParticles subclasses. Default value: 1.1

**radius**, float or np.ndarray: The particle radii, meters. If a NumPy ndarray is provided, then it must be 1D and have length equal to NumPart. Only for the FallingParticles subclasses. Default value: 0.0005

**rho**, float or np.ndarray: The particle density, kilograms per cubic meter. If a NumPy ndarray is provided, then it must be 1D and have length equal to NumPart. Only for the FallingParticles subclasses. Default value: 0.0005
