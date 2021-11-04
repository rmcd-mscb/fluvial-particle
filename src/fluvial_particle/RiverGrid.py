"""RiverGrid class module."""
import pathlib
import h5py
import numpy as np
import vtk
from vtk.util import numpy_support  # type:ignore


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
        self._read_2d_data()
        if track3d:
            self.track3d = 1
            self.vtksgrid3d = vtk.vtkStructuredGrid()
            self._read_3d_data()
            self.ns, self.nn, self.nz = self.vtksgrid3d.GetDimensions()
            self.nsc = self.ns - 1
            self.nnc = self.nn - 1
        else:
            self.track3d = 0
            self.ns, self.nn, self.nz = self.vtksgrid2d.GetDimensions()
            self.nsc = self.ns - 1
            self.nnc = self.nn - 1
        self._load_arrays()
        self._build_locators()

    def create_hdf5(self, dimtime, time, fname="cells.h5"):
        """Create HDF5 file for cell-centered results.

        Args:
            dimtime ([type]): [description]
            time ([type]): [description]
            fname ([type]): [description]

        Returns:
            [type]: [description]
        """
        vtkcoords = self.vtksgrid3d.GetPoints().GetData()
        coords = numpy_support.vtk_to_numpy(vtkcoords)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # ordering of the vtk grid points, keep the reshapes this way
        # so that paraview can read it and the cell-centered data
        # coherently; YOU ALREADY TRIED THE OTHER PERMUTATIONS
        x = x.reshape(self.nz, self.nn, self.ns)
        y = y.reshape(self.nz, self.nn, self.ns)
        z = z.reshape(self.nz, self.nn, self.ns)
        zeros = np.zeros((self.ns - 1,), dtype="f")
        arr = np.zeros((self.ns - 1, self.nn - 1, self.nz - 1), dtype="f")
        cells_h5 = h5py.File(fname, "w")
        grpg = cells_h5.create_group("grid")
        grp1 = cells_h5.create_group("cells1d")
        grp2 = cells_h5.create_group("cells2d")
        grp3 = cells_h5.create_group("cells3d")
        grpg.create_dataset("X", (self.nz, self.nn, self.ns), data=x)
        grpg.create_dataset("Y", (self.nz, self.nn, self.ns), data=y)
        grpg.create_dataset("Z", (self.nz, self.nn, self.ns), data=z)
        grpg.create_dataset("time", (dimtime, 1), data=time)
        grpg.create_dataset(
            "zeros", (self.ns - 1), data=zeros
        )  # for 1D viz in paraview
        for i in np.arange(dimtime):
            dname = f"fpc{i}"
            grp1.create_dataset(dname, (self.ns - 1), data=arr[:, 0, 0])
            grp2.create_dataset(dname, (self.ns - 1, self.nn - 1), data=arr[:, :, 0])
            grp3.create_dataset(
                dname, (self.ns - 1, self.nn - 1, self.nz - 1), data=arr
            )
            dname = f"tpc{i}"
            grp2.create_dataset(dname, (self.ns - 1, self.nn - 1), data=arr[:, :, 0])
            grp3.create_dataset(
                dname, (self.ns - 1, self.nn - 1, self.nz - 1), data=arr
            )
        return cells_h5

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

    def _read_2d_data(self):
        """Read 2D structured grid data file."""
        # Check that input file exists
        inputfile = pathlib.Path(self._fname2d)
        if not inputfile.exists():
            raise Exception(f"Cannot find 2D input file {inputfile}")
        # Read 2D grid
        reader2d = vtk.vtkStructuredGridReader()
        reader2d.SetFileName(self._fname2d)
        reader2d.SetOutput(self.vtksgrid2d)
        reader2d.Update()
        # Check for required field arrays
        a = self.vtksgrid2d.GetAttributesAsFieldData(0)  # 0 for point data
        names = [a.GetArrayName(i) for i in range(a.GetNumberOfArrays())]
        missing = [x for x in self.required_keys2d if x not in names]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} from the user parameter file")

    def _read_3d_data(self):
        """Read 3D structured grid data file."""
        # Check that input file exists
        inputfile = pathlib.Path(self._fname3d)
        if not inputfile.exists():
            raise Exception(f"Cannot find 3D input file {inputfile}")
        # Read 2D grid
        reader3d = vtk.vtkStructuredGridReader()
        reader3d.SetFileName(self._fname3d)
        reader3d.SetOutput(self.vtksgrid3d)
        reader3d.Update()
        # Check for required field arrays
        a = self.vtksgrid3d.GetAttributesAsFieldData(0)  # 0 for point data
        names = [a.GetArrayName(i) for i in range(a.GetNumberOfArrays())]
        missing = [x for x in self.required_keys3d if x not in names]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} from the user parameter file")

    @property
    def required_keys2d(self):
        return ("Elevation",
                "IBC",
                "ShearStress (magnitude)",
                "Velocity",
                "WaterSurfaceElevation"
                )

    @property
    def required_keys3d(self):
        return ("Velocity",)

    def update_velocity_fields(self, tidx):
        """Updates time-dependent velocity vtk arrays.

        Args:
            tidx ([type]): [description]
        """
        # want this to be callable from every time index, including 0
        # want it to support a different timestep than the particles timestep
        # can all of the time-dependent velocity data be stored in the same vtk file, e.g. along a new dimension?
        # or do we need to load a new file with every new time step?
        # the answer to these questions will determine how we implement

    def write_hdf5(self, obj, name, data):
        """Write cell arrays to HDF5 object.

        Args:
            obj ([type]): [description]
            name ([type]): [description]
            data ([type]): [description]
        """
        obj[name][...] = data

    def write_hdf5_xmf(
        self, filexmf, time, dims, names, attrnames, dtypes=None, center="Cell"
    ):
        """Body for cell-centered time series data.

        Args:
            filexmf ([type]): [description]
            time ([type]): [description]
            dims ([type]): [description]
            attrnames ([type]): [description]
            names ([type]): [description]
            dtypes ([type]): [description]
            center([type]): [description]
        """
        filexmf.write(
            f"""
            <Grid Name="t={time}" GridType="Uniform">
                <Time Value="{time}"/>
                <Topology Reference="XML">/Xdmf/Domain/Topology[@Name="Topo"]</Topology>
                <Geometry Reference="XML">/Xdmf/Domain/Geometry[@Name="Geo"]</Geometry>"""
        )
        if dtypes is not None:
            for i, j, k in zip(names, attrnames, dtypes):
                self.write_hdf5_xmf_attr(filexmf, dims, i, j, center, k)
        else:
            for i, j in zip(names, attrnames):
                self.write_hdf5_xmf_attr(filexmf, dims, i, j, center)
        filexmf.write(
            """
            </Grid>"""
        )

    def write_hdf5_xmf_attr(self, filexmf, dims, name, attrname, center, dtype="Float"):
        """[summary].

        Args:
            filexmf ([type]): [description]
            dims ([type]): [description]
            name ([type]): [description]
            attrname ([type]): [description]
            center ([type]): [description]
            dtype (str, optional): [description]. Defaults to "Float".
        """
        filexmf.write(
            f"""
            <Attribute Name="{attrname}" Center="{center}">
                <DataItem Dimensions="{" ".join(str(i) for i in dims)}" DataType="{dtype}" Format="HDF">
                    cells.h5:{name}
                </DataItem>
            </Attribute>"""
        )

    def write_hdf5_xmf_header1d(self, filexmf):
        """Header for cell-centered time series data.

        Args:
            filexmf ([type]): [description]
        """
        # Important! For whatever reason, dimensions of the xdmf Topology and Geometry files must be switched
        # relative to their order in both the input NumPy arrays and in the cell-centered xdmf body; can't
        # figure out why, but this is the only permutation that works.
        # You've tried the other permutations, don't mess with it
        ns, nn, nz = self.vtksgrid3d.GetDimensions()
        filexmf.write(
            f"""<Xdmf Version="3.0">
            <Domain>
                <Topology Name="Topo" TopologyType="PolyVertex" NodesPerElement="{self.ns - 1}"/>
                <Geometry Name="Geo" GeometryType="X_Y">
                    <DataItem ItemType="Hyperslab" Dimensions="1 1 {self.ns - 1}" Format="XML">
                        <DataItem Dimensions="3 3" Format="XML">
                            0 0 0
                            1 1 1
                            1 1 {self.ns - 1}
                        </DataItem>
                        <DataItem Name="X" Dimensions="{nz} {self.nn} {self.ns}" Format="HDF">
                            cells.h5:/grid/X
                        </DataItem>
                    </DataItem>
                    <DataItem Name="Y" Dimensions="{self.ns - 1}" Format="HDF">
                        cells.h5:/grid/zeros
                    </DataItem>
                </Geometry>
                <Grid Name="1D time series" GridType="Collection" CollectionType="Temporal">"""
        )

    def write_hdf5_xmf_header2d(self, filexmf):
        """Header for cell-centered time series data.

        Args:
            filexmf ([type]): [description]
        """
        # Important! For whatever reason, dimensions of the xdmf Topology and Geometry files must be switched
        # relative to their order in both the input NumPy arrays and in the cell-centered xdmf body; can't
        # figure out why, but this is the only permutation that works.
        # You've tried the other permutations, don't mess with it
        filexmf.write(
            f"""<Xdmf Version="3.0">
            <Domain>
                <Topology Name="Topo" TopologyType="2DSMesh" Dimensions="{self.nn} {self.ns}"/>
                <Geometry Name="Geo" GeometryType="X_Y">
                    <DataItem ItemType="Hyperslab" Dimensions="1 {self.nn} {self.ns}" Format="XML">
                        <DataItem Dimensions="3 3" Format="XML">
                            0 0 0
                            1 1 1
                            1 {self.nn} {self.ns}
                        </DataItem>
                        <DataItem Name="X" Dimensions="{self.nz} {self.nn} {self.ns}" Format="HDF">
                            cells.h5:/grid/X
                        </DataItem>
                    </DataItem>
                    <DataItem ItemType="Hyperslab" Dimensions="1 {self.nn} {self.ns}" Format="XML">
                        <DataItem Dimensions="3 3" Format="XML">
                            0 0 0
                            1 1 1
                            1 {self.nn} {self.ns}
                        </DataItem>
                        <DataItem Name="Y" Dimensions="{self.nz} {self.nn} {self.ns}" Format="HDF">
                            cells.h5:/grid/Y
                        </DataItem>
                    </DataItem>
                </Geometry>
                <Grid Name="2D time series" GridType="Collection" CollectionType="Temporal">"""
        )

    def write_hdf5_xmf_header3d(self, filexmf):
        """Header for cell-centered time series data.

        Args:
            filexmf ([type]): [description]
        """
        # Important! For whatever reason, dimensions of the xdmf Topology and Geometry files must be switched
        # relative to their order in both the input NumPy arrays and in the cell-centered xdmf body; can't
        # figure out why, but this is the only permutation that works.
        # You've tried the other permutations, don't mess with it
        filexmf.write(
            f"""<Xdmf Version="3.0">
            <Domain>
                <Topology Name="Topo" TopologyType="3DSMesh" Dimensions="{self.nz} {self.nn} {self.ns}"/>
                <Geometry Name="Geo" GeometryType="X_Y_Z">
                    <DataItem Name="X" Dimensions="{self.nz} {self.nn} {self.ns}" Format="HDF">
                        cells.h5:/grid/X
                    </DataItem>
                    <DataItem Name="Y" Dimensions="{self.nz} {self.nn} {self.ns}" Format="HDF">
                        cells.h5:/grid/Y
                    </DataItem>
                    <DataItem Name="Z" Dimensions="{self.nz} {self.nn} {self.ns}" Format="HDF">
                        cells.h5:/grid/Z
                    </DataItem>

                </Geometry>
                <Grid Name="3D time series" GridType="Collection" CollectionType="Temporal">"""
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
