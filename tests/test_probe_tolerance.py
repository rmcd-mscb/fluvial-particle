"""The probe pipeline must locate cells exactly, with no tolerance band at cell faces (issue #41).

VTK 9.7 started passing vtkProbeFilter's auto-computed tolerance through to the cell locator,
which assigned points within a few millimetres of a cell face to the neighbouring cell. That
flipped wet/dry checks at the bank and changed every 2D/3D regression fixture. RiverGrid now
sets a zero tolerance; this test fails on VTK 9.7 without it and passes on every version with it.
"""

import numpy as np
from vtk.util import numpy_support

from fluvial_particle.RiverGrid import RiverGrid
from tests.support import write_straight_channel


FIELD_MAP_2D = {
    "bed_elevation": "bed_elevation",
    "wet_dry": "wet_dry",
    "shear_stress": "shear_stress",
    "velocity": "velocity",
    "water_surface_elevation": "water_surface_elevation",
}
FIELD_MAP_3D = {"velocity": "velocity"}


def test_wet_dry_probe_is_exact_at_the_bank(tmp_path):
    """Points 1 mm either side of the last wet node line get the cell they are in, not a neighbour."""
    dy, half = 2.0, 20.0
    paths = write_straight_channel(tmp_path, length=100.0, width=100.0, dx=5.0, dy=dy, wet_halfwidth=half + 1e-6)
    offsets = np.array([1e-6, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2])
    # Cells touching a dry node are dry, so the wall is at |y| = half exactly.
    y_dry = np.concatenate([-half - offsets, half + offsets])
    y_wet = np.concatenate([-half + offsets, half - offsets])
    y = np.concatenate([y_dry, y_wet])
    x = np.full(y.size, 52.5)
    river = RiverGrid(0, paths[0], None, FIELD_MAP_2D, FIELD_MAP_3D)
    river.build_probe_filter(y.size)
    river.update_2d_pipeline(x, y)
    wet = numpy_support.vtk_to_numpy(river.probe2d.GetOutput().GetPointData().GetArray("CellWetDry"))
    expected = np.concatenate([np.zeros(y_dry.size), np.ones(y_wet.size)])
    np.testing.assert_array_equal(wet, expected, err_msg=f"y={y}\nwet={wet}")
