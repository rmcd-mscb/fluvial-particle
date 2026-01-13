"""VTP (VTK PolyData) writer for particle output."""

from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support


class VTPWriter:
    """Write particle data to VTK PolyData (.vtp) files.

    VTP files are natively supported by ParaView without any plugins,
    making them ideal for visualization of particle trajectories.
    """

    def __init__(self, output_dir: Path | str):
        """Initialize the VTP writer.

        Args:
            output_dir: Directory where VTP files will be written.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, particles, time: float, tidx: int) -> Path | None:
        """Write particle state to a VTP file.

        Args:
            particles: Particles instance with current state.
            time: Current simulation time in seconds.
            tidx: Time step index (used for filename).

        Returns:
            Path to the written VTP file, or None if no valid particles.
        """
        # Get valid (non-NaN) particle indices
        valid_mask = ~np.isnan(particles.x)
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return None

        # Create VTK points
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(n_valid)

        x_valid = particles.x[valid_mask]
        y_valid = particles.y[valid_mask]
        z_valid = particles.z[valid_mask]

        coords = np.column_stack([x_valid, y_valid, z_valid])
        vtk_coords = numpy_support.numpy_to_vtk(coords, deep=True)
        points.SetData(vtk_coords)

        # Create polydata with vertex cells
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        verts = vtk.vtkCellArray()
        for i in range(n_valid):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        polydata.SetVerts(verts)

        # Add scalar attributes
        self._add_scalar(polydata, "Depth", particles.depth[valid_mask])
        self._add_scalar(polydata, "BedElevation", particles.bedelev[valid_mask])
        self._add_scalar(polydata, "WaterSurfaceElevation", particles.wse[valid_mask])
        self._add_scalar(polydata, "HeightAboveBed", particles.htabvbed[valid_mask])
        self._add_scalar(polydata, "ShearStress", particles.shearstress[valid_mask])

        # Add integer attributes
        self._add_scalar(polydata, "CellIndex2D", particles.cellindex2d[valid_mask], dtype=np.int64)
        self._add_scalar(polydata, "CellIndex3D", particles.cellindex3d[valid_mask], dtype=np.int64)

        # Add velocity vector
        vel = np.column_stack([
            particles.velx[valid_mask],
            particles.vely[valid_mask],
            particles.velz[valid_mask],
        ])
        self._add_vector(polydata, "Velocity", vel)

        # Add time metadata to field data
        time_arr = vtk.vtkDoubleArray()
        time_arr.SetName("TimeValue")
        time_arr.SetNumberOfTuples(1)
        time_arr.SetValue(0, time)
        polydata.GetFieldData().AddArray(time_arr)

        # Write VTP file
        vtp_file = self.output_dir / f"particles_{tidx:04d}.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(vtp_file))
        writer.SetInputData(polydata)
        writer.SetDataModeToBinary()
        writer.Write()

        return vtp_file

    def _add_scalar(self, polydata: vtk.vtkPolyData, name: str, data: np.ndarray, dtype=None):
        """Add a scalar array to polydata point data.

        Args:
            polydata: VTK PolyData to add the array to.
            name: Name of the array.
            data: NumPy array of scalar values.
            dtype: Optional dtype to cast to before adding.
        """
        if dtype is not None:
            data = data.astype(dtype)
        arr = numpy_support.numpy_to_vtk(data, deep=True)
        arr.SetName(name)
        polydata.GetPointData().AddArray(arr)

    def _add_vector(self, polydata: vtk.vtkPolyData, name: str, data: np.ndarray):
        """Add a vector array to polydata point data.

        Args:
            polydata: VTK PolyData to add the array to.
            name: Name of the array.
            data: NumPy array of shape (n, 3) with vector components.
        """
        arr = numpy_support.numpy_to_vtk(data, deep=True)
        arr.SetName(name)
        polydata.GetPointData().AddArray(arr)
