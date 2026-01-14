"""Options file for fluvial particle model using time-varying VTS grids."""

from fluvial_particle.Particles import Particles

# Field name mappings for time-series output files
field_map_2d = {
    "bed_elevation": "Elevation[m]",
    "shear_stress": "Tausta",
    "velocity": "Velocity",
    "water_surface_elevation": "WaterSurf.[m]",
    # wet_dry omitted - will auto-compute from depth
}
field_map_3d = {
    "velocity": "Velocity",
}

# Time-dependent grid settings
time_dependent = True
file_pattern_2d = "./tests/data/time_series_straight/Result_2D_{}.vts"
file_pattern_3d = "./tests/data/time_series_straight/Result_3D_{}.vts"
grid_start_index = 2  # First file index (Result_*_2.vts)
grid_end_index = 6  # Last file index (Result_*_6.vts)
grid_dt = 1.0  # Time between grid files [seconds]
grid_interpolation = "linear"  # linear | nearest | hold

# For backward compatibility, also provide static file paths
# (used if time_dependent is False or not specified)
file_name_2d = "./tests/data/time_series_straight/Result_2D_2.vts"
file_name_3d = "./tests/data/time_series_straight/Result_3D_2.vts"

SimTime = 3.5  # Maximum simulation time [seconds] - within grid time range
dt = 0.1  # Simulation time step [seconds]
PrintAtTick = 1.0  # Print every PrintAtTick seconds

Track3D = 1  # 1 to use 3D velocity field, 0 to use 2D velocity field

NumPart = 20  # Number of particles to simulate per processor

# Starting locations: give tuple for exact point, or path to a checkpoint HDF5 file
StartLoc = (5, 0, 9.5)
startfrac = 0.5

# Particle type for the simulation
ParticleType = Particles  # noqa: F401

# Optional Particles keyword arguments
beta = (0.067, 0.067, 0.067)  # eddy viscosity coefficient
lev = 0.00025  # reach-averaged lateral eddy viscosity
min_depth = 0.02  # minimum depth cells may enter [meters]
vertbound = 0.01  # depth fraction that bounds particles at bed and water surface
