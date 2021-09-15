"""RiverVTKGrid class module."""
import vtk


class RiverVTKGrid:
    """A class of hydrodynamic data on a structured VTK grid."""

    def __init__(self, track3d):
        """[summary].

        Args:
            track3d ([type]): [description]
        """
        self.vtksgrid2d = vtk.vtkStructuredGrid()
        if track3d:
            self.track3D = 1
            self.vtksgrid3d = vtk.vtkStructuredGrid()
        else:
            self.track3D = 0

    def read_2d_data(self, filename):
        """[summary].

        Args:
            filename ([type]): [description]
        """
        # Assert filename???

        reader2d = vtk.vtkStructuredGridReader()
        reader2d.SetFileName(filename)
        reader2d.SetOutput(self.vtksgrid2d)
        reader2d.Update()

    def read_3d_data(self, filename):
        """[summary].

        Args:
            filename ([type]): [description]
        """
        # Assert filename???
        # Assert track3d?

        reader3d = vtk.vtkStructuredGridReader()
        reader3d.SetFileName(filename)
        reader3d.SetOutput(self.vtksgrid3d)
        reader3d.Update()

    def load_arrays(self):
        """[summary]."""
        # Get Elevation and WSE from 2D Grid
        self.WSE_2D = self.vtksgrid2d.GetPointData().GetScalars("WaterSurfaceElevation")
        self.Depth_2D = self.vtksgrid2d.GetPointData().GetScalars("Depth")
        self.Elevation_2D = self.vtksgrid2d.GetPointData().GetScalars("Elevation")
        self.Velocity_2D = self.vtksgrid2d.GetPointData().GetScalars(
            "Velocity (magnitude)"
        )
        self.IBC_2D = self.vtksgrid2d.GetPointData().GetScalars("IBC")
        self.VelocityVec2D = self.vtksgrid2d.GetPointData().GetVectors("Velocity")
        self.ShearStress2D = self.vtksgrid2d.GetPointData().GetScalars(
            "ShearStress (magnitude)"
        )
        # Get Velocity from 3D
        if self.track3d:
            self.VelocityVec3D = self.vtksgrid3d.GetPointData().GetScalars("Velocity")

    def build_locators(self):
        """[summary]."""
        self.CellLocator2D = vtk.vtkStaticCellLocator()
        self.CellLocator2D.SetDataSet(self.vtksgrid2d)
        self.CellLocator2D.BuildLocator()
        # CellLocator2D.SetNumberOfCellsPerBucket(5)
        # Build Static Cell Locators (thread-safe)
        if self.track3D:
            self.CellLocator3D = vtk.vtkStaticCellLocator()
            self.CellLocator3D.SetDataSet(self.vtksgrid3d)
            # CellLocator3D.SetNumberOfCellsPerBucket(5);
            # CellLocator3D.SetTolerance(0.000000001)
            self.CellLocator3D.BuildLocator()
