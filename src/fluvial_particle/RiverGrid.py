"""RiverGrid class module."""
import pathlib

import h5py
import numpy as np
import vtk
from vtk.util import numpy_support  # type:ignore

# from numba import jit


class RiverGrid:
    """A class of hydrodynamic data and tools defined on VTK structured grids."""

    def __init__(self, track3d, filename2d, filename3d=None):
        """Initialize instance of class RiverGrid.

        Args:
            track3d (int): 1 if 3D model run, 0 else
            filename2d (string): path to the input 2D VTK structured grid
            filename3d (string, optional): path to the input 2D VTK structured grid. Required for 3D simulations
        """
        self.vtksgrid2d = vtk.vtkStructuredGrid()
        self.vtksgrid3d = None
        self.fname2d = filename2d
        self._fname3d = None
        self.read_2d_data()
        if track3d:
            if filename3d is None:
                raise Exception(
                    "track3d is 1 but no filename provided for input 3D grid"
                )
            self.track3d = 1
            self.fname3d = filename3d
            self.vtksgrid3d = vtk.vtkStructuredGrid()
            self.read_3d_data()
            self.ns, self.nn, self.nz = self.vtksgrid3d.GetDimensions()
        else:
            self.track3d = 0
            self.ns, self.nn, self.nz = self.vtksgrid2d.GetDimensions()
        self.nsc = self.ns - 1
        self.nnc = self.nn - 1
        self.nzc = self.nz - 1
        self.process_arrays()

    def build_probe_filter(self, nparts, comm=None):
        """Build pipeline for probe filters (i.e. interpolation).

        Args:
            nparts (int): the number of input points to be probed
            comm (mpi4py object): MPI communicator, for parallel runs only
        """
        # Objects for 2d grid interpolation
        # Pipeline for VTK objects to have a view of data stored in NumPy arrays
        self.pt2d_np = np.zeros((nparts, 3))  # ordering required by vtk
        self.pt2d_vtk = numpy_support.numpy_to_vtk(self.pt2d_np)
        self.pt2d = vtk.vtkPoints()
        self.ptset2d = vtk.vtkPointSet()  # vtkPointSet() REQUIRES vtk>=9.1
        self.strategy2d = vtk.vtkCellLocatorStrategy()  # requires vtk>=9.0
        self.pt2d.SetData(self.pt2d_vtk)
        self.ptset2d.SetPoints(self.pt2d)
        # Build the probe
        self.probe2d = vtk.vtkProbeFilter()
        self.probe2d.SetInputData(self.ptset2d)
        self.probe2d.SetSourceData(self.vtksgrid2d)
        self.probe2d.SetFindCellStrategy(self.strategy2d)
        # Objects for 3d grid interpolation
        if self.track3d:
            self.pt3d_np = np.zeros((nparts, 3))
            self.pt3d_vtk = numpy_support.numpy_to_vtk(self.pt3d_np)
            self.pt3d = vtk.vtkPoints()
            self.pts3d = vtk.vtkPointSet()
            if comm is None:
                self.probe3d = vtk.vtkProbeFilter()
            else:
                self.probe3d = vtk.vtkPProbeFilter()
            strategy3d = vtk.vtkCellLocatorStrategy()
            self.pt3d.SetData(self.pt3d_vtk)
            self.pts3d.SetPoints(self.pt3d)
            self.probe3d.SetInputData(self.pts3d)
            self.probe3d.SetSourceData(self.vtksgrid3d)
            self.probe3d.SetFindCellStrategy(strategy3d)

    def create_hdf5(self, nprints, time, fname="cells.h5"):
        """Create HDF5 file for cell-centered results.

        Args:
            nprints (int): number of printing time steps
            time (NumPy ndarray): array of print times
            fname (string): file name of output HDF5 file

        Returns:
            cells_h5: new open HDF5 file object
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
        grpg.create_dataset("time", (nprints, 1), data=time)
        grpg.create_dataset("zeros", (nsc,), data=zeros)
        for i in np.arange(nprints):
            dname = f"fpc{i}"
            grp1.create_dataset(dname, (nsc,), data=arr[:, 0, 0])
            grp2.create_dataset(dname, (nsc, nnc), data=arr[:, :, 0])
            if self.track3d:
                grp3.create_dataset(dname, (nsc, nnc, nzc), data=arr)
        return cells_h5

    def out_of_grid(self, px, py, idx=None):
        """Check if any points in the probe filter pipeline are out of the 2D domain.

        Args:
            px (NumPy ndarray): x coordinates of new points
            py (NumPy ndarray): y coordinates of new points
            idx (NumPy ndarray, optional): active indices in px & py. Defaults to None.

        Returns:
            (NumPy ndarray): dtype=bool, True for indices of points out of the domain, else False
        """
        self.update_2d_pipeline(px, py, idx)
        out = self.probe2d.GetOutput()

        # Interpolation on cell-centered ordered integer array gives cell index number
        cellidxvtk = out.GetPointData().GetArray("CellIndex")
        cellidx = numpy_support.vtk_to_numpy(cellidxvtk)

        # Points in river boundary cells are considered out of the domain
        idxss = np.searchsorted(self.boundarycells, cellidx)
        bndrycells = np.equal(self.boundarycells[idxss], cellidx)

        # Points that have wandered outside the 2D grid are out of domain
        valid = self.probe2d.GetValidPoints()
        outofgrid = np.full(bndrycells.shape, fill_value=False)
        if out.GetNumberOfPoints() != valid.GetNumberOfTuples():
            name = self.probe2d.GetValidPointMaskArrayName()
            msk = out.GetPointData().GetArray(name)
            msk_np = numpy_support.vtk_to_numpy(msk)
            outofgrid[msk_np < 1] = True

        # Return True for points that satisfy either condition
        outparts = np.logical_or(bndrycells, outofgrid)

        return outparts

    def process_arrays(self):
        """Add required / delete unneeded arrays from 2D and 3D structured grids."""
        # Remove unneeded point data arrays from 2D grid
        ptdata = self.vtksgrid2d.GetPointData()
        names = [ptdata.GetArrayName(i) for i in range(ptdata.GetNumberOfArrays())]
        reqd = self.required_keys2d
        for x in names:
            if x not in reqd:
                ptdata.RemoveArray(x)

        # Add two cell-centered int arrays to 2D grid; index array and IBC array for checking particle wetness
        numcells = self.vtksgrid2d.GetNumberOfCells()
        cidx = vtk.vtkIntArray()
        cidx.SetNumberOfComponents(1)
        cidx.SetNumberOfTuples(numcells)
        cidx.SetName("CellIndex")
        cibc = vtk.vtkIntArray()
        cibc.SetNumberOfComponents(1)
        cibc.SetNumberOfTuples(numcells)
        cibc.Fill(1)  # Set all cells to wet by default
        cibc.SetName("CellIBC")
        ibc = ptdata.GetArray("IBC")
        for cellidx in range(numcells):
            cidx.SetValue(cellidx, cellidx)  # Set cell index
            # Check wetness cell-by-cell
            cell = self.vtksgrid2d.GetCell(cellidx)
            cellpts = cell.GetPointIds()
            for ptidx in range(cellpts.GetNumberOfIds()):
                # If any pts bounding cell have IBC < 1, then set to dry
                if ibc.GetTuple(cellpts.GetId(ptidx))[0] < 1:
                    cibc.SetValue(cellidx, 0)
                    break
        self.vtksgrid2d.GetCellData().AddArray(cibc)
        self.vtksgrid2d.GetCellData().AddArray(cidx)
        ptdata.RemoveArray("IBC")  # no longer needed

        # Set boundarycells array, particles that enter these cells will be deactivated
        # On structured grid, always assumes river flows in or out through the i=1, i=imax faces, and
        # not the j=1,j=jmax faces (although this could be easily added)
        firstcells = np.arange(0, self.nsc * (self.nnc - 1) + 1, self.nsc)
        lastcells = np.arange(self.nsc - 1, self.nsc * self.nnc, self.nsc)
        self.boundarycells = np.union1d(firstcells, lastcells)

        if self.track3d:
            # Remove unneeded point data arrays from 3D grid
            ptdata = self.vtksgrid3d.GetPointData()
            names = [ptdata.GetArrayName(i) for i in range(ptdata.GetNumberOfArrays())]
            reqd = self.required_keys3d
            for x in names:
                if x not in reqd:
                    ptdata.RemoveArray(x)
            # Remove velocity vector data from 2D grid, won't be using it
            self.vtksgrid2d.GetPointData().RemoveArray("Velocity")
            # Add cell-centered index array to 3D grid
            numcells = self.vtksgrid3d.GetNumberOfCells()
            cidx3 = vtk.vtkIntArray()
            cidx3.SetNumberOfComponents(1)
            cidx3.SetNumberOfTuples(numcells)
            cidx3.SetName("CellIndex")
            for i in range(numcells):
                cidx3.SetTuple(i, [i])
            self.vtksgrid3d.GetCellData().AddArray(cidx3)

    def read_2d_data(self):
        """Read 2D structured grid data file."""
        # Read 2D grid
        reader2d = vtk.vtkStructuredGridReader()
        reader2d.SetFileName(self.fname2d)
        reader2d.SetOutput(self.vtksgrid2d)
        reader2d.Update()
        # Check for required field arrays defined at the grid points
        ptdata = self.vtksgrid2d.GetPointData()
        names = [ptdata.GetArrayName(i) for i in range(ptdata.GetNumberOfArrays())]
        missing = [x for x in self.required_keys2d if x not in names]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} array(s) from the input 2D grid")

    def read_3d_data(self):
        """Read 3D structured grid data file."""
        # Read 2D grid
        reader3d = vtk.vtkStructuredGridReader()
        reader3d.SetFileName(self.fname3d)
        reader3d.SetOutput(self.vtksgrid3d)
        reader3d.Update()
        # Check for required field arrays defined at the grid points
        ptdata = self.vtksgrid3d.GetPointData()
        names = [ptdata.GetArrayName(i) for i in range(ptdata.GetNumberOfArrays())]
        missing = [x for x in self.required_keys3d if x not in names]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} array from the input 3D grid")

    def reconstruct_filter_pipeline(self, nparts):
        """Reconstruct VTK probe filter pipeline objects.

        Args:
            nparts (int): the number of input points to be probed
        """
        self.pt2d_np = np.zeros((nparts, 3))
        self.pt2d_vtk = numpy_support.numpy_to_vtk(self.pt2d_np)
        self.pt2d.Reset()
        self.pt2d.SetData(self.pt2d_vtk)
        self.ptset2d.Initialize()
        self.ptset2d.SetPoints(self.pt2d)
        self.probe2d.SetInputData(self.ptset2d)
        if self.track3d:
            self.pt3d_np = np.zeros((nparts, 3))
            self.pt3d_vtk = numpy_support.numpy_to_vtk(self.pt3d_np)
            self.pt3d.Reset()
            self.pt3d.SetData(self.pt3d_vtk)
            self.pts3d.Initialize()
            self.pts3d.SetPoints(self.pt3d)
            self.probe3d.SetInputData(self.pts3d)

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

    def update_2d_pipeline(self, px, py, idx=None):
        """Update the 2D VTK probe filter pipeline.

        Args:
            px (NumPy ndarray): x coordinates of new points
            py (NumPy ndarray): y coordinates of new points
            idx (NumPy ndarray, optional): active indices in px & py. Defaults to None.
        """
        if idx is None:
            self.pt2d_np[:, 0] = px
            self.pt2d_np[:, 1] = py
        else:
            self.pt2d_np[:, 0] = px[idx]
            self.pt2d_np[:, 1] = py[idx]
        self.pt2d.Modified()
        self.probe2d.Update()

    def update_3d_pipeline(self, px, py, pz, idx=None):
        """Update the 3D VTK probe filter pipeline.

        Args:
            px (NumPy ndarray): x coordinates of new points
            py (NumPy ndarray): y coordinates of new points
            pz (NumPy ndarray): z coordinates of new points
            idx (NumPy ndarray, optional): active indices in px, py & pz. Defaults to None.
        """
        if idx is None:
            self.pt3d_np[:, 0] = px
            self.pt3d_np[:, 1] = py
            self.pt3d_np[:, 2] = pz
        else:
            self.pt3d_np[:, 0] = px[idx]
            self.pt3d_np[:, 1] = py[idx]
            self.pt3d_np[:, 2] = pz[idx]
        self.pt3d.Modified()
        self.probe3d.Update()

    def update_velocity_fields(self, tidx):
        """Updates time-dependent field arrays on VTK structured grids.

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
            obj (h5py object): opened HDF5 file to write data to
            name (str): key of the data to be written
            data (NumPy ndarray): data to be written
        """
        obj[name][...] = data

    def write_hdf5_xmf(
        self, filexmf, time, dims, names, attrnames, dtypes=None, center="Cell"
    ):
        """Body for cell-centered time series data.

        Args:
            filexmf (file): open file to write
            time (float): current simulation time
            dims (tuple): integer values describing dimensions of the grid
            names (list of strings): paths to datasets from the root directory in the HDF5 file
            attrnames (list of strings): descriptive names corresponding to names
            dtypes (list of strings): data types corresponding to names (either Float or Int)
            center(str): Node for node-centered data, Cell for cell-centered data
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
        """Write data sets as attributes to the XDMF file.

        Args:
            filexmf (file): open file to write
            dims (tuple): integer values describing dimensions of the grid
            name (str): path to dataset from the root directory in the HDF5 file
            attrname (str): descriptive name of dataset
            center(str): Node for node-centered data, Cell for cell-centered data
            dtype (str): dataset type, defaults to Float
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
            filexmf (file): open file to write
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
            filexmf (file): open file to write
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
            filexmf (file): open file to write
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
            filexmf (file): open file to write
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
        if not isinstance(values, np.ndarray):
            raise TypeError("boundarycells.setter: wrong type, must be NumPy ndarray")
        if values.ndim != 1:
            raise ValueError(
                "boundarycells.setter: ndims must equal 1 for use in np.sortedsearch()"
            )
        if not is_sorted(values):
            print(
                "RiverGrid: boundarycells.setter array must be sorted to use binary search; sorting in-place"
            )
            values.sort()
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
            raise TypeError("nn.setter must be int")
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
            raise TypeError("nnc.setter must be int")
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
            raise TypeError("ns.setter must be int")
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
            raise TypeError("nsc.setter must be int")
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
            raise TypeError("nz.setter must be int")
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
            raise TypeError("nzc.setter must be int")
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
        if not isinstance(values, int):
            raise TypeError("track3d.setter must be int")
        if values < 0 or values > 1:
            raise ValueError("track3d.setter must be 0 or 1")
        self._track3d = values


# @jit(nopython=True)
def is_sorted(arr):
    """Using Numba, an efficient check that a 1D NumPy array is sorted in increasing order.

    https://stackoverflow.com/questions/47004506/check-if-a-numpy-array-is-sorted

    Args:
        arr ([type]): [description]

    Returns:
        [type]: [description]
    """
    for i in range(arr.size - 1):
        if arr[i + 1] < arr[i]:
            return False
    return True
