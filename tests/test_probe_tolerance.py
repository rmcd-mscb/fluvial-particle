"""The probe pipeline must locate cells exactly, with no tolerance band at cell faces (issue #41).

``vtkProbeFilter`` hands its auto-computed tolerance (0.1% of the largest cell diagonal) to the
cell locator. VTK 9.7's locators honour it where earlier ones ignored it, so points within a few
millimetres of a cell face were assigned to the neighbouring cell. That flipped wet/dry checks at
the bank and failed every 2D/3D regression case. ``RiverGrid._configure_probe`` sets a zero
tolerance and keeps the find-cell strategy below 9.7; the bank and meander tests here fail on 9.7
without the zero tolerance and pass on 9.6 and 9.7 with it.
"""

import warnings

import numpy as np
import pytest
import vtk
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

MEANDER_2D = "tests/data/Result_FM_MEander_1_long_2D1.vtk"
MEANDER_3D = "tests/data/Result_FM_MEander_1_long_3D1_new.vtk"
FIELD_MAP_2D_VTK = {
    "bed_elevation": "Elevation",
    "wet_dry": "IBC",
    "shear_stress": "ShearStress (magnitude)",
    "velocity": "Velocity",
    "water_surface_elevation": "WaterSurfaceElevation",
}
FIELD_MAP_3D_VTK = {"velocity": "Velocity"}

PRE_97 = (vtk.vtkVersion.GetVTKMajorVersion(), vtk.vtkVersion.GetVTKMinorVersion()) < (9, 7)


def valid_mask(probe):
    """Return the probe's valid-point mask as a boolean array."""
    arr = probe.GetOutput().GetPointData().GetArray(probe.GetValidPointMaskArrayName())
    return numpy_support.vtk_to_numpy(arr).astype(bool)


def point_array(probe, name):
    """Return a named point-data array of the probe output."""
    return numpy_support.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(name))


@pytest.mark.parametrize("cls", [vtk.vtkProbeFilter, vtk.vtkPProbeFilter], ids=["serial", "mpi"])
def test_configure_probe_sets_zero_tolerance_and_gates_the_strategy(cls):
    """Both probe classes get a zero tolerance; the deprecated strategy is attached only below 9.7."""
    probe = cls()
    # On 9.7 GetFindCellStrategy is deprecated and always returns None, so the only observable sign
    # of the strategy being attached there is the DeprecationWarning its constructor emits.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        RiverGrid._configure_probe(probe)
    np.testing.assert_equal(probe.GetTolerance(), 0.0)
    assert not probe.GetComputeTolerance()
    if PRE_97:
        assert probe.GetFindCellStrategy() is not None


def test_configure_probe_raises_if_vtk_ignores_the_zero_tolerance():
    """A VTK that clamps or ignores the zero tolerance fails at grid load rather than drifting results."""

    class Stubborn(vtk.vtkProbeFilter):
        def GetTolerance(self):  # noqa: N802  (overrides the VTK method)
            return 1e-3

    with pytest.raises(RuntimeError, match="did not accept a zero probe tolerance"):
        RiverGrid._configure_probe(Stubborn())


def test_wet_dry_probe_is_exact_at_the_bank(tmp_path):
    """Points 1 um to 1 cm either side of the last wet node line get the cell they are in, not a neighbour."""
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
    wet = point_array(river.probe2d, "CellWetDry")
    expected = np.concatenate([np.zeros(y_dry.size), np.ones(y_wet.size)])
    np.testing.assert_array_equal(wet, expected, err_msg=f"y={y}\nwet={wet}")


def _cell_contains(grid, cell_id, point, tol=1e-6):
    """True if ``point`` lies inside cell ``cell_id`` (parametric coordinates within [0, 1])."""
    cell = grid.GetCell(int(cell_id))
    closest, pcoords = [0.0] * 3, [0.0] * 3
    weights = [0.0] * cell.GetNumberOfPoints()
    sub_id, dist2 = vtk.reference(0), vtk.reference(0.0)
    inside = cell.EvaluatePosition(list(point), closest, sub_id, pcoords, dist2, weights)
    return inside == 1 and min(pcoords) >= -tol and max(pcoords) <= 1.0 + tol


@pytest.mark.parametrize("comm", [None, object()], ids=["serial", "mpi"])
def test_3d_probe_assigns_the_containing_hex_near_bed_and_surface(comm):
    """On a curvilinear grid, points 5 mm off the bed or surface land in the hex that contains them.

    An axis-aligned synthetic channel barely exercises this (a handful of the 400 points on VTK 9.7
    with the default tolerance); the meander fixture's skewed hexes put about a tenth of the near-bed
    points in the neighbouring cell, so it is the sharper guard.
    """
    # build_probe_filter only tests comm for None; vtkPProbeFilter runs serially without a controller.
    river = RiverGrid(1, MEANDER_2D, MEANDER_3D, FIELD_MAP_2D_VTK, FIELD_MAP_3D_VTK)
    pts = numpy_support.vtk_to_numpy(river.vtksgrid2d.GetPoints().GetData())
    pd = river.vtksgrid2d.GetPointData()
    depth = numpy_support.vtk_to_numpy(pd.GetArray("water_surface_elevation")) - numpy_support.vtk_to_numpy(
        pd.GetArray("bed_elevation")
    )
    rng = np.random.RandomState(0)
    deep = np.flatnonzero(depth > 0.3)
    xy = pts[rng.choice(deep, 400, replace=False)][:, :2] + rng.uniform(-0.2, 0.2, (400, 2))
    river.build_probe_filter(len(xy), comm=comm)
    assert type(river.probe3d).__name__ == ("vtkProbeFilter" if comm is None else "vtkPProbeFilter")
    river.update_2d_pipeline(xy[:, 0], xy[:, 1])
    bed = point_array(river.probe2d, "bed_elevation")
    wse = point_array(river.probe2d, "water_surface_elevation")
    for label, z in (("bed", bed + 5e-3), ("surface", wse - 5e-3)):
        river.update_3d_pipeline(xy[:, 0], xy[:, 1], z)
        cell_id = point_array(river.probe3d, "CellIndex")
        valid = valid_mask(river.probe3d)
        bad = [i for i in np.flatnonzero(valid) if not _cell_contains(river.vtksgrid3d, cell_id[i], (*xy[i], z[i]))]
        assert not bad, f"{len(bad)} points 5 mm off the {label} assigned to a hex that does not contain them"


def test_points_exactly_on_faces_edges_and_corners_stay_valid(tmp_path):
    """A zero tolerance must not make points exactly on cell faces, grid edges, bed, or surface invalid.

    Only validity and out_of_grid are asserted, not the cell index: a point exactly on a node line
    may land in either adjacent cell depending on the VTK version.
    """
    p2, p3 = write_straight_channel(tmp_path, length=100.0, width=100.0, dx=5.0, dy=2.0, depth=2.0, nz=11)
    river = RiverGrid(1, p2, p3, FIELD_MAP_2D, FIELD_MAP_3D)
    # interior face, node, node line, x=0 edge, x=100 edge, y edges, corners
    xy = np.array([
        [52.5, -20.0],
        [50.0, -19.0],
        [50.0, -20.0],
        [0.0, 0.0],
        [100.0, 0.0],
        [52.5, -50.0],
        [52.5, 50.0],
        [0.0, -50.0],
        [100.0, 50.0],
    ])
    river.build_probe_filter(len(xy))
    out = river.out_of_grid(xy[:, 0], xy[:, 1])
    assert valid_mask(river.probe2d).all()
    # The first and last cells along x are boundary cells and count as out of grid; everything else is in.
    np.testing.assert_array_equal(out, [False, False, False, True, True, False, False, True, True])
    xyz = np.array([
        [0.0, -50.0, 0.0],
        [100.0, 50.0, 2.0],
        [52.5, -20.0, 0.0],
        [52.5, -20.0, 0.2],
        [52.5, -20.0, 2.0],
        [52.5, -20.0, 1.0],
    ])
    river.build_probe_filter(len(xyz))
    river.update_3d_pipeline(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    assert valid_mask(river.probe3d).all()
