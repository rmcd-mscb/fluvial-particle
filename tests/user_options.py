"""Options file for fluvial particle model."""
# Required simulation options
file_name_3d = "data/Result_FM_MEander_1_long_3D1_new.vtk"  # path to 3D mesh
file_name_2d = "data/Result_FM_MEander_1_long_2D1.vtk"  # path to 2D mesh
SimTime = 100.0  # maximum simulation time [seconds]
dt = 0.25  # simulation time step [seconds]
Track3D = 1  # 1 to use 3D velocity field, 0 to use 2D velocity field
NumPart = 1000  # Number of particles to simulate (per processor)
PrintAtTick = 1.0  # Print every PrintAtTick seconds
# StartLoc = (490.0, -4965.0, 530.0)  # Start Locations
StartLoc = (6.14, 9.09, 10.3)

# Declare particle type for the simulation
ParticleType = LarvalTopParticles  # noqa

# Optional keyword arguments
min_depth = 0.02  # minimum depth cells may enter [meters]
lev = 0.00025  # reach Lateral Eddy Viscosity
beta = (0.067, 0.067, 0.067)  # eddy viscosity coefficient
vertbound = 0.01  # depth fraction that bounds particles at bed and water surface

# Add keyword arguments for Particles subclasses here
period = 60.0
amp = 0.4
