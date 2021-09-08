"""Settings for PartivatTrack runs."""
__author__ = "rmcd"


#  259299.0 == 72 Hours  3 Days
#   86400.0 == 24 Hours
#   21600.0 ==  6 Hours
#   10800.0 ==  3 Hours
#    3600.0 ==  1 Hour

# Prod
SimTime = 1000.0
dt = 0.25
avg_depth = 0.5
avg_bed_shearstress = 6
avg_shear_dev = 0.004
min_depth = 0.02

# vertical movement
# vert_type = int
# vert_type = 0: vertConstDepth
# vert_type = 1: vert_sinusoid
# vert_type = 2: vert_sinusoid_bottom
# vert_type = 3: vert_sinusoid_surface
# vert_type = 4: vert_sawtooth
# vert_type = 5: vert_random_walk

vert_type = 5

# DispersionType = 1 Local ku*h/6
# DispersionType = 2 Local ku*h/6 + LEV
# DispersionType = 3 Reach avg ku*h/6
# DispersionType = 4 Reach avg ku*h/6 + LEV
DispersionType = 2

LEV = 0.0025

beta_x = 0.067
beta_y = 0.067
beta_z = 0.067

NumPart = 1000

# A Tick = seconds * delta so  7200 ticks ==  1 hour
# A Tick = seconds * delta so 14400 ticks ==  2 hour
# A Tick = seconds * delta so 28800 ticks ==  4 hour 6 per Day

# A Tick = seconds * delta so 57600 ticks ==  8 hour 3 per Day
# A Tick = seconds * delta so  2400 ticks == 20 mins 3 per hour
CheckAtTick = 2

# A Tick = seconds * delta so   20 ticks ==  10 seconds
# A Tick = seconds * delta so 1200 ticks == 600 seconds == 10 minutes
PrintAtTick = 4

delta = 0.1

# yo-yo paramters
amplitude = 1.0
period = 60.0
min_elev = 0.01

# Start Locations
StartLoc = (6.14, 9.09, 10.3)
# StartLoc = (4645,-3163,774)
# StartLoc = (1819,-4963,776)
# StartLoc = (289,-236,765)

Track2D = 0
Track3D = 1

# Source Files
# A2DFile = 'NoStrmLnCurv_185cms2d1.vtk'
# A3DFile = 'NoStrmLnCurv_185cms3d1.vtk'
A2DFile = "Result_2D_1.vtk"
A3DFile = "Result_3D_1.vtk"


out_2d_dir = "Sim2D"
out_3d_dir = "Sim3D"
# out_part = 'NoStrmLnCurv_185cms2d1_part.vtk'
out_part = "Result_2D_Part.vtk"

file_name_3da = "/home/aprescott/fluvparticle/data/Result_FM_MEander_1_long_3D1.vtk"
file_name_2da = "/home/aprescott/fluvparticle/data/Result_FM_MEander_1_long_2D1.vtk"
out_dir = r"/home/aprescott/fluvparticle/output"


def printtimes():
    """[summary].

    Returns:
        [type]: [description]
    """
    step = CheckAtTick * delta
    return range(int(step), int(SimTime) + 1, int(step))


if __name__ == "__main__":
    print("CheckPoint at"), printtimes()
