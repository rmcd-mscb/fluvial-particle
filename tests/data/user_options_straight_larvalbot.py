"""Options file for fluvial particle model."""

from fluvial_particle.LarvalParticles import LarvalBotParticles

# Field name mappings from standard names to model-specific names
field_map_2d = {
    "bed_elevation": "Elevation",
    "wet_dry": "IBC",
    "shear_stress": "ShearStress (magnitude)",
    "velocity": "Velocity",
    "water_surface_elevation": "WaterSurfaceElevation",
}
field_map_3d = {
    "velocity": "Velocity",
}

# Paths to 3D and 2D mesh files
file_name_3d = "./tests/data/Result_straight_3d_1_new.vtk"
file_name_2d = "./tests/data/Result_straight_2d_1.vtk"

SimTime = 60.0  # maximum simulation time [seconds]
dt = 0.25  # simulation time step [seconds]
PrintAtTick = 20.0  # Print every PrintAtTick seconds

Track3D = 1  # 1 to use 3D velocity field, 0 to use 2D velocity field

NumPart = 20  # Number of particles to simulate per processor

# Starting locations: give tuple for exact point, or path to a checkpoint HDF5 file
StartLoc = (5.0, 0.0, 10.1)
startfrac = 0.5

# Particle type for the simulation
ParticleType = LarvalBotParticles  # noqa: F401

# Optional Particles keyword arguments
beta = (0.067, 0.067, 0.067)  # eddy viscosity coefficient
lev = 0.00025  # reach-averaged lateral eddy viscosity
min_depth = 0.02  # minimum depth cells may enter [meters]
vertbound = 0.01  # depth fraction that bounds particles at bed and water surface
amp = 0.2
period = 60.0
