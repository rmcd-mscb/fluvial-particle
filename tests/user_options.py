"""Options file for fluvial particle model."""
file_name_3da = "~/Meander_269cms_Result_3D_lev_250_1.vtk"
file_name_2da = "~/Meander_269cms_Result_2D_lev_250_1.vtk"


# Prod
SimTime = 129600.0
dt = 0.25
min_depth = 0.02

LEV = 0.25

beta_x = 0.067
beta_y = 0.067
beta_z = 0.067

NumPart = 10000

# Print every PrintAtTick seconds
PrintAtTick = 800.0

# Start Locations
StartLoc = (490.0, -4965.0, 530.0)
# StartLoc = (6.14, 9.09, 10.3)

Track3D = 1
