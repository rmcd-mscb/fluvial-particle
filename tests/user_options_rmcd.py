"""Options file for fluvial particle model."""
# Paths to 3D and 2D mesh files
file_name_3d = "./tests/data/Result_FM_MEander_1_long_3D1_new.vtk"
file_name_2d = "./tests/data/Result_FM_MEander_1_long_2D1.vtk"

SimTime = 300.0  # maximum simulation time [seconds]
dt = 0.25  # simulation time step [seconds]
PrintAtTick = 1.0  # Print every PrintAtTick seconds

Track3D = 1  # 1 to use 3D velocity field, 0 to use 2D velocity field

NumPart = 20  # Number of particles to simulate per processor

# Starting locations: give tuple for exact point, or path to a checkpoint HDF5 file
# StartLoc = "tests/test/particles.h5"
# StartIdx = 45  # optional time slice index to pull data from, defaults to -1
# StartLoc = (490.0, -4965.0, 530.0)
# StartLoc = (6.14, 9.09, 10.3)
StartLoc = "./tests/data/varsrc.csv"

# Particle type for the simulation
ParticleType = VarSrcParticles  # noqa
radius = 0.0001  # optional keyword argument for FallingParticles subclass

# Optional Particles keyword arguments
beta = (0.067, 0.067, 0.067)  # eddy viscosity coefficient
lev = 0.00025  # reach-averaged lateral eddy viscosity
min_depth = 0.02  # minimum depth cells may enter [meters]
vertbound = 0.01  # depth fraction that bounds particles at bed and water surface
