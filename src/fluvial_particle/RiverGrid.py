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
        self.vtksgrid3d = None
        self.fname2d = filename2d
        self._fname3d = None
        self._read_2d_data()
        if track3d:
            self.track3d = 1
            self.fname3d = filename3d
            self.vtksgrid3d = vtk.vtkStructuredGrid()
            self._read_3d_data()
            self.ns, self.nn, self.nz = self.vtksgrid3d.GetDimensions()
        else:
            self.track3d = 0
            self.ns, self.nn, self.nz = self.vtksgrid2d.GetDimensions()
        self.nsc = self.ns - 1
        self.nnc = self.nn - 1
        self.nzc = self.nz - 1
        self._load_arrays()
        self._build_locators()

        # On structured grid, always assumes river flows in or out through the i=1, i=imax faces,
        # not the j=1,j=jmax faces (although this could be added)
        firstcells = np.arange(0, self.nsc * (self.nnc - 1) + 1, self.nsc)
        lastcells = np.arange(self.nsc - 1, self.nsc * self.nnc, self.nsc)
        self.boundarycells = np.union1d(firstcells, lastcells)

    def create_hdf5(self, dimtime, time, fname="cells.h5"):
        """Create HDF5 file for cell-centered results.

        Args:
            dimtime ([type]): [description]
            time ([type]): [description]
            fname ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.track3d:
            vtkcoords = self.vtksgrid3d.GetPoints().GetData()
        else:
            vtkcoords = self.vtksgrid2d.GetPoints().GetData()
        coords = numpy_support.vtk_to_numpy(vtkcoords)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # ordering of the vtk grid points, keep the reshapes this way
        # so that paraview can read it and the cell-centered data
        # coherently; YOU ALREADY TRIED THE OTHER PERMUTATIONS
        ns = self.ns
        nn = self.nn
        nz = self.nz
        nsc = self.nsc
        nnc = self.nnc
        nzc = self.nzc
        x = x.reshape(nz, nn, ns)
        y = y.reshape(nz, nn, ns)
        z = z.reshape(nz, nn, ns)
        zeros = np.zeros((nsc,), dtype="f")
        arr = np.zeros((nsc, nnc, nzc), dtype="f")
        cells_h5 = h5py.File(fname, "w")
        grpg = cells_h5.create_group("grid")
        grp1 = cells_h5.create_group("cells1d")
        grp2 = cells_h5.create_group("cells2d")
        if self.track3d:
            grp3 = cells_h5.create_group("cells3d")
        grpg.create_dataset("X", (nz, nn, ns), data=x)
        grpg.create_dataset("Y", (nz, nn, ns), data=y)
        grpg.create_dataset("Z", (nz, nn, ns), data=z)
        grpg.create_dataset("time", (dimtime, 1), data=time)
        grpg.create_dataset("zeros", (nsc,), data=zeros)
        for i in np.arange(dimtime):
            dname = f"fpc{i}"
            grp1.create_dataset(dname, (nsc,), data=arr[:, 0, 0])
            grp2.create_dataset(dname, (nsc, nnc), data=arr[:, :, 0])
            if self.track3d:
                grp3.create_dataset(dname, (nsc, nnc, nzc), data=arr)
            dname = f"tpc{i}"
            grp2.create_dataset(dname, (nsc, nnc), data=arr[:, :, 0])
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
        # Read 2D grid
        reader2d = vtk.vtkStructuredGridReader()
        reader2d.SetFileName(self.fname2d)
        reader2d.SetOutput(self.vtksgrid2d)
        reader2d.Update()
        # Check for required field arrays
        a = self.vtksgrid2d.GetAttributesAsFieldData(0)  # 0 for point data
        names = [a.GetArrayName(i) for i in range(a.GetNumberOfArrays())]
        missing = [x for x in self.required_keys2d if x not in names]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} array from the input 2D grid")

    def _read_3d_data(self):
        """Read 3D structured grid data file."""
        # Read 2D grid
        reader3d = vtk.vtkStructuredGridReader()
        reader3d.SetFileName(self.fname3d)
        reader3d.SetOutput(self.vtksgrid3d)
        reader3d.Update()
        # Check for required field arrays
        a = self.vtksgrid3d.GetAttributesAsFieldData(0)  # 0 for point data
        names = [a.GetArrayName(i) for i in range(a.GetNumberOfArrays())]
        missing = [x for x in self.required_keys3d if x not in names]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} array from the input 3D grid")

    @property
    def required_keys2d(self):
        """Array names required in the input 2D grid.

        Returns:
            tuple
        """
        return (
            "Elevation",
            "IBC",
            "ShearStress (magnitude)",
            "Velocity",
            "WaterSurfaceElevation",
        )

    @property
    def required_keys3d(self):
        """Array names required in the input 3D grid.

        Returns:
            tuple
        """
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
        filexmf.write(
            f"""<Xdmf Version="3.0">
            <Domain>
                <Topology Name="Topo" TopologyType="PolyVertex" NodesPerElement="{self.nsc}"/>
                <Geometry Name="Geo" GeometryType="X_Y">
                    <DataItem ItemType="Hyperslab" Dimensions="1 1 {self.nsc}" Format="XML">
                        <DataItem Dimensions="3 3" Format="XML">
                            0 0 0
                            1 1 1
                            1 1 {self.nsc}
                        </DataItem>
                        <DataItem Name="X" Dimensions="{self.nz} {self.nn} {self.ns}" Format="HDF">
                            cells.h5:/grid/X
                        </DataItem>
                    </DataItem>
                    <DataItem Name="Y" Dimensions="{self.nsc}" Format="HDF">
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

    # Properties

    @property
    def boundarycells(self):
        """Get inflow/outflow boundary cells.

        Returns:
            [type]: [description]
        """
        return self._boundarycells

    @boundarycells.setter
    def boundarycells(self, values):
        """Set inflow/outflow boundary cells.

        Args:
            values ([type]): [description]
        """
        assert isinstance(values, np.ndarray), TypeError(  # noqa:S101
            "boundarycells.setter: wrong type, must be NumPy ndarray, ndims=1"
        )
        assert values.ndim == 1, ValueError(  # noqa: S101
            "boundarycells.setter: ndims must equal 1 for use in np.sortedsearch()"
        )
        self._boundarycells = values

    @property
    def fname2d(self):
        """Get 2d grid input filename.

        Returns:
            [type]: [description]
        """
        return self._fname2d

    @fname2d.setter
    def fname2d(self, values):
        """Set 2d grid input filename.

        Args:
            values ([type]): [description]
        """
        # Check that input file exists
        inputfile = pathlib.Path(values)
        if not inputfile.exists():
            raise Exception(f"Cannot find 2D input file {inputfile}")
        self._fname2d = values

    @property
    def fname3d(self):
        """Get 3d grid input filename.

        Returns:
            [type]: [description]
        """
        return self._fname3d

    @fname3d.setter
    def fname3d(self, values):
        """Set 3d grid input filename.

        Args:
            values ([type]): [description]
        """
        # Check that input file exists
        inputfile = pathlib.Path(values)
        if not inputfile.exists():
            raise Exception(f"Cannot find 3D input file {inputfile}")
        self._fname3d = values

    @property
    def nn(self):
        """Get nn, number of stream-normal points that define the grids.

        Returns:
            [type]: [description]
        """
        return self._nn

    @nn.setter
    def nn(self, values):
        """Set nn, number of stream-normal points that define the grids.

        Args:
            values ([type]): [description]
        """
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nn.setter wrong type")
        # for file writing reasons, must be >= 1
        if values < 1:
            values = 1
        self._nn = values

    @property
    def nnc(self):
        """Get nnc, number of stream-normal cells defined by the grids.

        Returns:
            [type]: [description]
        """
        return self._nnc

    @nnc.setter
    def nnc(self, values):
        """Set nnc, number of stream-normal cells defined by the grids.

        Args:
            values ([type]): [description]
        """
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nnc.setter wrong type")
        # for file writing reasons, must be >= 1
        if values < 1:
            values = 1
        self._nnc = values

    @property
    def ns(self):
        """Get ns, number of stream-wise points that define the grids.

        Returns:
            [type]: [description]
        """
        return self._ns

    @ns.setter
    def ns(self, values):
        """Set ns, number of stream-wise points that define the grids.

        Args:
            values ([type]): [description]
        """
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("ns.setter wrong type")
        # for file writing reasons, must be >= 1
        if values < 1:
            values = 1
        self._ns = values

    @property
    def nsc(self):
        """Get nsc, number of stream-wise cells defined by the grids.

        Returns:
            [type]: [description]
        """
        return self._nsc

    @nsc.setter
    def nsc(self, values):
        """Set nsc, number of stream-wise cells defined by the grids.

        Args:
            values ([type]): [description]
        """
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nsc.setter wrong type")
        # for file writing reasons, must be >= 1
        if values < 1:
            values = 1
        self._nsc = values

    @property
    def nz(self):
        """Get nz, number of vertical points that define the grids.

        Returns:
            [type]: [description]
        """
        return self._nz

    @nz.setter
    def nz(self, values):
        """Set nz, number of vertical points that define the grids.

        Args:
            values ([type]): [description]
        """
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nz.setter wrong type")
        # for file writing reasons, must be >= 1
        if values < 1:
            values = 1
        self._nz = values

    @property
    def nzc(self):
        """Get nzc, number of vertical cells defined by the grids.

        Returns:
            [type]: [description]
        """
        return self._nzc

    @nzc.setter
    def nzc(self, values):
        """Set nzc, number of vertical cells defined by the grids.

        Args:
            values ([type]): [description]
        """
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nzc.setter wrong type")
        # for file writing reasons, must be >= 1
        if values < 1:
            values = 1
        self._nzc = values

    @property
    def track3d(self):
        """Get track3d.

        Returns:
            [type]: [description]
        """
        return self._track3d

    @track3d.setter
    def track3d(self, values):
        """Set track3d.

        Args:
            values ([type]): [description]
        """
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("track3d.setter wrong type")
        self._track3d = values
