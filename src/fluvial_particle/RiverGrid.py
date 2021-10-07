"""RiverGrid class module."""
import vtk


class RiverGrid:
    """A class of hydrodynamic data on a structured VTK grid."""

    def __init__(self, track3d, filename2d, filename3d=None):
        """Initialize instance of class RiverGrid.

        Args:
            track3d ([type]): [description]
            filename2d ([type]): [description]
            filename3d ([type], optional): [description]. Defaults to None.
        """
        self.vtksgrid2d = vtk.vtkStructuredGrid()
        self._fname2d = filename2d
        self._fname3d = filename3d
        self._read_2d_data
        if track3d:
            self.track3d = 1
            self.vtksgrid3d = vtk.vtkStructuredGrid()
            if filename3d is not None:
                self._read_3d_data
            else:
                print("no 3d filename provided")
        else:
            self.track3d = 0
        self._load_arrays
        self._build_locators

    @property
    def _build_locators(self):
        """Build Static Cell Locators (thread-safe)."""
        self.CellLocator2D = vtk.vtkStaticCellLocator()
        self.CellLocator2D.SetDataSet(self.vtksgrid2d)
        self.CellLocator2D.BuildLocator()
        # CellLocator2D.SetNumberOfCellsPerBucket(5)
        if self.track3d:
            self.CellLocator3D = vtk.vtkStaticCellLocator()
            self.CellLocator3D.SetDataSet(self.vtksgrid3d)
            # CellLocator3D.SetNumberOfCellsPerBucket(5);
            # CellLocator3D.SetTolerance(0.000000001)
            self.CellLocator3D.BuildLocator()

    @property
    def _load_arrays(self):
        """Load 2D and 3D structured grid arrays."""
        self.WSE_2D = self.vtksgrid2d.GetPointData().GetScalars("WaterSurfaceElevation")
        self.Depth_2D = self.vtksgrid2d.GetPointData().GetScalars("Depth")
        self.Elevation_2D = self.vtksgrid2d.GetPointData().GetScalars("Elevation")
        self.IBC_2D = self.vtksgrid2d.GetPointData().GetScalars("IBC")
        self.VelocityVec2D = self.vtksgrid2d.GetPointData().GetVectors("Velocity")
        self.ShearStress2D = self.vtksgrid2d.GetPointData().GetScalars(
            "ShearStress (magnitude)"
        )
        # Get Velocity from 3D
        if self.track3d:
            self.VelocityVec3D = self.vtksgrid3d.GetPointData().GetScalars("Velocity")

    @property
    def _read_2d_data(self):
        """Read 2D structured grid data file."""
        # Assert filename???

        reader2d = vtk.vtkStructuredGridReader()
        reader2d.SetFileName(self._fname2d)
        reader2d.SetOutput(self.vtksgrid2d)
        reader2d.Update()
        self.ns, self.nn, self.nz = self.vtksgrid2d.GetDimensions()
        self.nsc = self.ns - 1
        self.nnc = self.nn - 1
        # output2d = reader2d.GetOutput()
        # scalar_range = output2d.GetScalarRange()

    @property
    def _read_3d_data(self):
        """Read 3D structured grid data file."""
        # Assert filename???
        if self.track3d:
            reader3d = vtk.vtkStructuredGridReader()
            reader3d.SetFileName(self._fname3d)
            reader3d.SetOutput(self.vtksgrid3d)
            reader3d.Update()
            # output3d = reader3d.GetOutput()
            # scalar_range = output3d.GetScalarRange()

    def write_hdf5(self, obj, name, data, idx):
        """Write cell arrays to HDF5 object.

        Args:
            obj ([type]): [description]
            name ([type]): [description]
            data ([type]): [description]
            idx ([type]): [description]
        """
        obj[name][idx, :, :] = data.reshape(self.ns - 1, self.nn - 1)

    def write_hdf5_xmf(self, filexmf, time, nsteps, attrname, name, idx):
        """Body for cell-centered time series data.

        Args:
            filexmf ([type]): [description]
            time ([type]): [description]
            nsteps ([type]): [description]
            attrname ([type]): [description]
            name ([type]): [description]
            idx ([type]): [description]
        """
        filexmf.write(
            f"""
            <Grid GridType="Uniform">
                <Time Value="{time}"/>
                <Topology Reference="XML">/Xdmf/Domain/Topology[@Name="Topo"]</Topology>
                <Geometry Reference="XML">/Xdmf/Domain/Geometry[@Name="Geo"]</Geometry>
                <Attribute Name="{attrname}" AttributeType="Scalar" Center="Cell">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {self.ns - 1} {self.nn - 1}" Format="XML">
                        <DataItem Dimensions="3 3" Format="XML">
                            {idx} 0 0
                            1 1 1
                            1 {self.ns - 1} {self.nn - 1}
                        </DataItem>
                        <DataItem Dimensions="{nsteps} {self.ns - 1} {self.nn - 1}" Format="HDF">cells.h5:{name}</DataItem>
                    </DataItem>
                </Attribute>
            </Grid>"""
        )

    def write_hdf5_xmf_header(self, filexmf):
        """Header for cell-centered time series data.

        Args:
            filexmf ([type]): [description]
        """
        # Important! For whatever reason, dimensions of the xdmf Topology and Geometry files must be switched
        # relative to their order in both the input NumPy arrays and in the cell-centered xdmf body; can't
        # figure out why, but this is the only permutation that works.
        # You've tried the other permutations, don't mess with it
        filexmf.write(
            """<Xdmf Version="3.0">
            <Domain>
                <Topology Name="Topo" TopologyType="2DSMesh" Dimensions="11 601"/>
                <Geometry Name="Geo" GeometryType="X_Y">
                    <DataItem Name="X" Dimensions="11 601" Format="HDF">
                        cells.h5:/X
                    </DataItem>
                    <DataItem Name="Y" Dimensions="11 601" Format="HDF">
                        cells.h5:/Y
                    </DataItem>
                </Geometry>
                <Grid GridType="Collection" CollectionType="Temporal">"""
        )

    def write_hdf5_xmf_footer(self, filexmf):
        """Footer for cell-centered time series data.

        Args:
            filexmf ([type]): [description]
        """
        filexmf.write(
            """
                </Grid>
            </Domain>
        </Xdmf>
        """
        )
