"""RiverGrid class module."""

import pathlib

import h5py
import numpy as np
import vtk
from vtk.util import numpy_support  # type: ignore[import]


# from numba import jit

# Standard internal field names used by fluvial-particle
# Required fields must be provided in field_map_2d
REQUIRED_FIELDS_2D = (
    "bed_elevation",
    "shear_stress",
    "velocity",
    "water_surface_elevation",
)

# Optional fields - if not provided, they will be computed internally
OPTIONAL_FIELDS_2D = ("wet_dry",)

# All standard 2D fields (required + optional)
STANDARD_FIELDS_2D = REQUIRED_FIELDS_2D + OPTIONAL_FIELDS_2D

STANDARD_FIELDS_3D = ("velocity",)

# Default minimum depth threshold for computing wet_dry (meters)
DEFAULT_MIN_DEPTH = 0.02


class RiverGrid:
    """A class of hydrodynamic data and tools defined on VTK structured grids."""

    def __init__(self, track3d, filename2d, filename3d=None, field_map_2d=None, field_map_3d=None, min_depth=None):
        """Initialize instance of class RiverGrid.

        Args:
            track3d (int): 1 if 3D model run, 0 else
            filename2d (str): path to the input 2D grid. Supported formats:
                - .vtk: VTK legacy structured grid format
                - .vts: VTK XML structured grid format (recommended for large files)
                - .npz: NumPy compressed archive format
                See the docstring of the read_2d_data() method for additional details.
            filename3d (str, optional): path to the input 3D grid. Required for 3D simulations.
                Supports the same formats as filename2d.
            field_map_2d (dict): Mapping from standard field names to model-specific names for
                the 2D grid. Required keys: bed_elevation, shear_stress, velocity,
                water_surface_elevation. Optional key: wet_dry (computed from depth if not provided).
                Example: {"bed_elevation": "Elevation", ...}
            field_map_3d (dict): Mapping from standard field names to model-specific names for
                the 3D grid. Required keys: velocity. Example: {"velocity": "Velocity"}
            min_depth (float, optional): Minimum depth threshold for computing wet_dry if not
                provided in field_map_2d. Cells with depth <= min_depth are considered dry.
                Defaults to 0.02 meters.

        Raises:
            ValueError: track3d is 1 but no filename provided for input 3D grid.
            ValueError: field_map_2d is missing required standard field names.
            ValueError: field_map_3d is missing required standard field names.
        """
        # Validate field mappings
        if field_map_2d is None:
            raise ValueError("field_map_2d is required")
        missing_2d = [k for k in REQUIRED_FIELDS_2D if k not in field_map_2d]
        if missing_2d:
            raise ValueError(f"field_map_2d is missing required keys: {missing_2d}")
        self.field_map_2d = field_map_2d

        # Check if wet_dry needs to be computed
        self._compute_wet_dry = "wet_dry" not in field_map_2d
        self._min_depth = min_depth if min_depth is not None else DEFAULT_MIN_DEPTH

        if field_map_3d is None:
            raise ValueError("field_map_3d is required")
        missing_3d = [k for k in STANDARD_FIELDS_3D if k not in field_map_3d]
        if missing_3d:
            raise ValueError(f"field_map_3d is missing required keys: {missing_3d}")
        self.field_map_3d = field_map_3d

        self.vtksgrid2d = vtk.vtkStructuredGrid()
        self.vtksgrid3d = None
        self.fname2d = filename2d
        self._fname3d = None
        self.read_2d_data()
        if track3d:
            if filename3d is None:
                raise ValueError("track3d is 1 but no filename provided for input 3D grid")
            self.track3d = 1
            self.fname3d = filename3d
            self.vtksgrid3d = vtk.vtkStructuredGrid()
            self.read_3d_data()
            dims = [0, 0, 0]
            self.vtksgrid3d.GetDimensions(dims)
            self.ns, self.nn, self.nz = dims
        else:
            self.track3d = 0
            dims = [0, 0, 0]
            self.vtksgrid2d.GetDimensions(dims)
            self.ns, self.nn, self.nz = dims
        self.nsc = self.ns - 1
        self.nnc = self.nn - 1
        self.nzc = self.nz - 1
        self.process_arrays()

    def _apply_field_mapping(self, vtkgrid, field_map):
        """Rename arrays in a VTK grid from model-specific names to standard names.

        Args:
            vtkgrid: VTK structured grid object
            field_map (dict): Mapping from standard names to model-specific names
        """
        ptdata = vtkgrid.GetPointData()
        # Create reverse mapping: model_name -> standard_name
        reverse_map = {v: k for k, v in field_map.items()}
        # Rename arrays
        for i in range(ptdata.GetNumberOfArrays()):
            arr = ptdata.GetArray(i)
            if arr is not None:
                model_name = arr.GetName()
                if model_name in reverse_map:
                    arr.SetName(reverse_map[model_name])

    def _compute_wet_dry_from_depth(self):
        """Compute wet_dry field from depth (water_surface_elevation - bed_elevation).

        Cells with depth > min_depth are considered wet (1), otherwise dry (0).
        This method is called when wet_dry is not provided in field_map_2d.
        """
        ptdata = self.vtksgrid2d.GetPointData()

        # Get bed elevation and water surface elevation arrays
        bed_elev = ptdata.GetArray("bed_elevation")
        wse = ptdata.GetArray("water_surface_elevation")

        if bed_elev is None or wse is None:
            raise ValueError("Cannot compute wet_dry: bed_elevation or water_surface_elevation not found")

        # Convert to numpy arrays for computation
        bed_elev_np = numpy_support.vtk_to_numpy(bed_elev)
        wse_np = numpy_support.vtk_to_numpy(wse)

        # Compute depth and wet_dry mask
        depth = wse_np - bed_elev_np
        wet_dry_np = (depth > self._min_depth).astype(np.float64)

        # Create VTK array and add to grid
        wet_dry_vtk = numpy_support.numpy_to_vtk(wet_dry_np)
        wet_dry_vtk.SetName("wet_dry")
        ptdata.AddArray(wet_dry_vtk)

        print(
            f"Computed wet_dry from depth (min_depth={self._min_depth}m): "
            f"{int(wet_dry_np.sum())} wet / {len(wet_dry_np)} total points"
        )

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

    def create_hdf5(self, nprints, time, fname="cells.h5", **dset_kwargs):
        """Create HDF5 file for cell-centered results.

        Args:
            nprints (int): number of printing time steps
            time (NumPy ndarray): array of print times
            fname (str): file name of output HDF5 file
            **dset_kwargs (dict): HDF5 dataset keyword arguments, e.g. compression filter # noqa: E501

        Returns:
            h5py file object: the newly created and open HDF5 file
        """
        vtkcoords = self.vtksgrid3d.GetPoints().GetData() if self.track3d else self.vtksgrid2d.GetPoints().GetData()
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
        grpg.create_dataset("X", (nz, nn, ns), data=x, **dset_kwargs)
        grpg.create_dataset("Y", (nz, nn, ns), data=y, **dset_kwargs)
        grpg.create_dataset("Z", (nz, nn, ns), data=z, **dset_kwargs)
        grpg.create_dataset("time", (nprints, 1), data=time, **dset_kwargs)
        grpg.create_dataset("zeros", (nsc,), data=zeros, **dset_kwargs)
        for i in np.arange(nprints):
            dname = f"fpc{i}"
            grp1.create_dataset(dname, (nsc,), data=arr[:, 0, 0], **dset_kwargs)
            grp2.create_dataset(dname, (nsc, nnc), data=arr[:, :, 0], **dset_kwargs)
            if self.track3d:
                grp3.create_dataset(dname, (nsc, nnc, nzc), data=arr, **dset_kwargs)
        return cells_h5

    def out_of_grid(self, px, py, idx=None):
        """Check if any points in the probe filter pipeline are out of the 2D domain.

        Args:
            px (NumPy ndarray): x coordinates of new points
            py (NumPy ndarray): y coordinates of new points
            idx (NumPy ndarray, optional): active indices in px & py. Defaults to None.

        Returns:
            ndarray(bool): True for indices of points out of the domain, else False
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

        return np.logical_or(bndrycells, outofgrid)

    def postprocess(self, output_directory, n_prints, globalnparts, **dset_kwargs):
        """Write XDMF files and cumulative cell counters, must be executed in serial.

        Args:
            output_directory (str): path to output directory
            n_prints (int): total number of printing time steps
            globalnparts (int): number of particles across all processors
            **dset_kwargs (dict): HDF5 dataset keyword arguments, e.g. compression filter # noqa: E501
        """
        # Open Particles HDF5 file for printing steps and cell locations
        parts_h5 = h5py.File(f"{output_directory}//particles.h5", "r")

        with pathlib.Path(f"{output_directory}//cells_onedim.xmf").open("w", encoding="utf-8") as cells1d_xmf:
            cells2d_xmf = pathlib.Path(f"{output_directory}//cells_twodim.xmf").open("w", encoding="utf-8")

            self.write_hdf5_xmf_header1d(cells1d_xmf)
            self.write_hdf5_xmf_header2d(cells2d_xmf)

            # Create cells HDF5 file and arrays
            grpc = parts_h5["coordinates"]
            grpp = parts_h5["properties"]
            time = grpc["time"]
            nsc = self.nsc
            num2dcells = self.vtksgrid2d.GetNumberOfCells()
            cells_h5 = self.create_hdf5(n_prints, time, f"{output_directory}//cells.h5", **dset_kwargs)
            numpartin2dcell = np.zeros(num2dcells, dtype=np.int64)
            # totpartincell = np.zeros(num2dcells, dtype=np.int64)
            numpartin1dcell = np.zeros(nsc, dtype=np.int64)

            if self.track3d:
                num3dcells = self.vtksgrid3d.GetNumberOfCells()
                cells3d_xmf = pathlib.Path(f"{output_directory}//cells_threedim.xmf").open("w", encoding="utf-8")
                self.write_hdf5_xmf_header3d(cells3d_xmf)
                numpartin3dcell = np.zeros(num3dcells, dtype=np.int64)

            # For every printing time loop, we load the particles data, sum the cell-centered counter arrays,
            # write the arrays to the cells HDF5, and write metadata to the XDMF files
            gen = [t for t in time if not np.isnan(t)]
            for i in range(len(gen)):
                t = gen[i].item(0)  # this returns a python scalar, for use in f-strings

                cell2d = grpp["cellidx2d"][i, :]
                numpartin1dcell[:] = 0
                numpartin2dcell[:] = 0
                np.add.at(numpartin1dcell, cell2d[cell2d >= 0] % nsc, 1)
                # np.add.at(totpartincell, cell2d[cell2d >= 0], 1)
                np.add.at(numpartin2dcell, cell2d[cell2d >= 0], 1)
                if self.track3d:
                    cell3d = grpp["cellidx3d"][i, :]
                    numpartin3dcell[:] = 0
                    np.add.at(numpartin3dcell, cell3d[cell3d >= 0], 1)

                # dims, name, and attrname must be passed to write_hdf5_xmf as iterable objects
                # dtypes too, but it is optional (defaults to "Float")
                name = [[]]
                attrname = [[]]
                name[0] = f"/cells1d/fpc{i}"
                attrname[0] = "FractionalParticleCount"
                data = numpartin1dcell / globalnparts
                self.write_hdf5(cells_h5, name[0], data)
                dims = (self.ns - 1,)
                self.write_hdf5_xmf(cells1d_xmf, t, dims, name, attrname, center="Node")

                name = [[]]  # , []]
                attrname = [[]]  # , []]
                dtypes = [[]]  # , []]
                name[0] = f"/cells2d/fpc{i}"
                attrname[0] = "FractionalParticleCount"
                dtypes[0] = "Float"
                dims = (self.ns - 1, self.nn - 1)
                data = (numpartin2dcell / globalnparts).reshape(dims)
                self.write_hdf5(cells_h5, name[0], data)
                """ # Total particle count is not accurately computed in this way
            # it only sums particle positions at printing time steps, not all simulation steps
            name[1] = f"/cells2d/tpc{i}"
            attrname[1] = "TotalParticleCount"
            dtypes[1] = "Int"
            dims = (self.ns - 1, self.nn - 1)
            data = totpartincell.reshape(dims)
            self.write_hdf5(cells_h5, name[1], data) """
                self.write_hdf5_xmf(cells2d_xmf, t, dims, name, attrname, dtypes)

                if self.track3d:
                    name = [[]]
                    attrname = [[]]
                    name[0] = f"/cells3d/fpc{i}"
                    attrname[0] = "FractionalParticleCount"
                    dims = (self.ns - 1, self.nn - 1, self.nz - 1)
                    data = (numpartin3dcell / globalnparts).reshape(dims)
                    self.write_hdf5(cells_h5, name[0], data)
                    self.write_hdf5_xmf(cells3d_xmf, t, dims, name, attrname)

            # Finalize xmf file writing
            self.write_hdf5_xmf_footer(cells1d_xmf)
            self.write_hdf5_xmf_footer(cells2d_xmf)
            if self.track3d:
                self.write_hdf5_xmf_footer(cells3d_xmf)
                cells3d_xmf.close()
        cells2d_xmf.close()
        cells_h5.close()
        parts_h5.close()

    def process_arrays(self):
        """Add required / delete unneeded arrays from 2D and 3D structured grids."""
        # Remove unneeded point data arrays from 2D grid
        ptdata = self.vtksgrid2d.GetPointData()
        names = [ptdata.GetArrayName(i) for i in range(ptdata.GetNumberOfArrays())]
        reqd = self.required_keys2d
        for x in names:
            if x not in reqd:
                ptdata.RemoveArray(x)

        # Add two cell-centered int arrays to 2D grid; index array and wet_dry array for checking particle wetness
        numcells = self.vtksgrid2d.GetNumberOfCells()
        cidx = vtk.vtkIntArray()
        cidx.SetNumberOfComponents(1)
        cidx.SetNumberOfTuples(numcells)
        cidx.SetName("CellIndex")
        cell_wet_dry = vtk.vtkIntArray()
        cell_wet_dry.SetNumberOfComponents(1)
        cell_wet_dry.SetNumberOfTuples(numcells)
        cell_wet_dry.Fill(1)  # Set all cells to wet by default
        cell_wet_dry.SetName("CellWetDry")
        wet_dry = ptdata.GetArray("wet_dry")
        for cellidx in range(numcells):
            cidx.SetValue(cellidx, cellidx)  # Set cell index
            # Check wetness cell-by-cell
            cell = self.vtksgrid2d.GetCell(cellidx)
            cellpts = cell.GetPointIds()
            for ptidx in range(cellpts.GetNumberOfIds()):
                # If any pts bounding cell have wet_dry < 1, then set to dry
                if wet_dry.GetTuple(cellpts.GetId(ptidx))[0] < 1:
                    cell_wet_dry.SetValue(cellidx, 0)
                    break
        self.vtksgrid2d.GetCellData().AddArray(cell_wet_dry)
        self.vtksgrid2d.GetCellData().AddArray(cidx)
        ptdata.RemoveArray("wet_dry")  # no longer needed

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
            self.vtksgrid2d.GetPointData().RemoveArray("velocity")
            # Add cell-centered index array to 3D grid
            numcells = self.vtksgrid3d.GetNumberOfCells()
            cidx3 = vtk.vtkIntArray()
            cidx3.SetNumberOfComponents(1)
            cidx3.SetNumberOfTuples(numcells)
            cidx3.SetName("CellIndex")
            for i in range(numcells):
                cidx3.SetTuple(i, [i])
            self.vtksgrid3d.GetCellData().AddArray(cidx3)

    def _validate_and_apply_field_mapping_2d(self):
        """Validate required fields exist and apply field mapping for 2D VTK grids."""
        ptdata = self.vtksgrid2d.GetPointData()
        names = [ptdata.GetArrayName(i) for i in range(ptdata.GetNumberOfArrays())]
        model_names = list(self.field_map_2d.values())
        missing = [x for x in model_names if x not in names]
        if len(missing) > 0:
            raise ValueError(f"Missing {missing} array(s) from the input 2D grid")
        self._apply_field_mapping(self.vtksgrid2d, self.field_map_2d)
        if self._compute_wet_dry:
            self._compute_wet_dry_from_depth()

    def read_2d_data(self):
        """Read 2D structured grid data file.

        Loads 2D data onto a VTK structured grid. The structured grid can be directly supplied as a .vtk file,
        a .vts (VTK XML) file, or as a collection of 2D arrays in a NumPy .npz file. The filename is read from
        the self.fname2d variable, saved during class initialization via the filename2d argument.

        The field_map_2d dict (provided at initialization) maps standard field names to model-specific names
        in the input file. After reading, arrays are renamed to standard names for internal use.

        Supported formats:
            - .vtk: VTK legacy structured grid format
            - .vts: VTK XML structured grid format (recommended for large files)
            - .npz: NumPy compressed archive format

        If a .npz file is supplied, then each expected field is a 2D array of the same shape. The (x, y) coordinates
        of every node that defines the grid are supplied via the x & y arguments. The remaining fields are all defined
        at the grid points. The npz file must have been saved using the following keyword arguments:

            - x: x coordinates of the grid
            - y: y coordinates of the grid
            - elev: topographic elevation
            - ibc: indicates whether node is wet (1) or dry (0), optional - computed from depth if missing
            - shear: shear stress magnitude
            - vx: x-component of fluid velocity
            - vy: y-component of fluid velocity
            - wse: water surface elevation
            - z (optional): z coordinates of the grid. Defaults to constant 0.
            - vz (optional): z-component of fluid velocity. Defaults to constant 0.
        """
        suffix = pathlib.Path(self.fname2d).suffix.lower()
        if suffix == ".vtk":
            reader2d = vtk.vtkStructuredGridReader()
            reader2d.SetFileName(self.fname2d)
            reader2d.SetOutput(self.vtksgrid2d)
            reader2d.Update()
            self._validate_and_apply_field_mapping_2d()
        elif suffix == ".vts":
            reader2d = vtk.vtkXMLStructuredGridReader()
            reader2d.SetFileName(self.fname2d)
            reader2d.Update()
            self.vtksgrid2d.ShallowCopy(reader2d.GetOutput())
            self._validate_and_apply_field_mapping_2d()
        elif suffix == ".npz":
            # Read 2D grid arrays from NumPy .npz file and convert to VTK grid
            # Note: .npz format uses fixed internal names, not the field_map
            npzfile = np.load(self.fname2d)
            # Required fields for npz (ibc is optional - will be computed from depth if missing)
            reqd = ["x", "y", "elev", "shear", "vx", "vy", "wse"]
            names = npzfile.files
            missing = [x for x in reqd if x not in names]
            if len(missing) > 0:
                raise ValueError(f"Missing {missing} array(s) from the input 2D grid npz file")
            x = npzfile["x"]
            dims = x.shape
            y = npzfile["y"]
            z = npzfile["z"] if "z" in names else np.zeros(dims)
            elev = npzfile["elev"]
            has_ibc = "ibc" in names
            ibc = npzfile["ibc"] if has_ibc else None
            shear = npzfile["shear"]
            vx = npzfile["vx"]
            vy = npzfile["vy"]
            vz = npzfile["vz"] if "vz" in names else np.zeros(dims)
            wse = npzfile["wse"]

            # make sure they're all the same shape and 2D
            ll = [x, y, z, elev, shear, vx, vy, vz, wse]
            if ibc is not None:
                ll.append(ibc)
            if not all(a.shape == dims for a in ll):
                raise Exception("input arrays in the 2D grid npz file must all be the same shape")
            if not len(dims) == 2:
                raise Exception("input arrays in the 2D grid npz file must be 2D")

            # ravel the arrays
            x = x.ravel()
            y = y.ravel()
            z = z.ravel()
            elev = elev.ravel()
            if ibc is not None:
                ibc = ibc.ravel()
            shear = shear.ravel()
            vx = vx.ravel()
            vy = vy.ravel()
            vz = vz.ravel()
            wse = wse.ravel()

            # make the coordinates
            ptdata = np.stack([x, y, z]).T
            vptdata = numpy_support.numpy_to_vtk(ptdata)
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(x.size)
            pts.SetData(vptdata)

            # make the grid
            grid = vtk.vtkStructuredGrid()
            grid.SetDimensions((dims[1], dims[0], 1))
            grid.SetPoints(pts)

            # combine the velocity components
            vel = np.stack([vx, vy, vz]).T

            # add required fields to the grid using standard names
            for arr, name in [
                (elev, "bed_elevation"),
                (shear, "shear_stress"),
                (vel, "velocity"),
                (wse, "water_surface_elevation"),
            ]:
                vtkarr = numpy_support.numpy_to_vtk(arr)
                vtkarr.SetName(name)
                grid.GetPointData().AddArray(vtkarr)

            # add wet_dry if provided in npz file
            if ibc is not None:
                vtkarr = numpy_support.numpy_to_vtk(ibc)
                vtkarr.SetName("wet_dry")
                grid.GetPointData().AddArray(vtkarr)

            # save to the class variable
            self.vtksgrid2d = grid

            # Compute wet_dry from depth if not provided in npz file
            if ibc is None:
                self._compute_wet_dry_from_depth()
        else:
            raise TypeError(
                f"{pathlib.Path(self.fname2d).suffix} file type not supported for input 2D grid; "
                f"expected .vtk, .vts, or .npz"
            )

    def read_3d_data(self):
        """Read 3D structured grid data file.

        Loads 3D data onto a VTK structured grid. The structured grid can be directly supplied as a .vtk file,
        a .vts (VTK XML) file, or as a collection of 3D arrays in a NumPy .npz file. The filename is read from
        the self.fname3d variable, saved during class initialization via the filename3d argument.

        The field_map_3d dict (provided at initialization) maps standard field names to model-specific names
        in the input file. After reading, arrays are renamed to standard names for internal use.

        Supported formats:
            - .vtk: VTK legacy structured grid format
            - .vts: VTK XML structured grid format (recommended for large files)
            - .npz: NumPy compressed archive format

        If a .npz file is supplied, then each expected field is a 3D array of the same shape. The (x, y, z) coordinates
        of every node that defines the grid are supplied via the x, y, & z arguments. The 3D fluid velocity is the other
        expected field, and it is defined at the grid points. The npz file must have been saved using the following
        keyword arguments:

            - x: x coordinates of the grid
            - y: y coordinates of the grid
            - z: z coordinates of the grid
            - vx: x-component of fluid velocity
            - vy: y-component of fluid velocity
            - vz: z-component of fluid velocity
        """
        suffix = pathlib.Path(self.fname3d).suffix.lower()
        if suffix == ".vtk":
            # Read 3D grid from VTK legacy structured grid format
            reader3d = vtk.vtkStructuredGridReader()
            reader3d.SetFileName(self.fname3d)
            reader3d.SetOutput(self.vtksgrid3d)
            reader3d.Update()
            # Check for required field arrays (using model-specific names from field_map)
            ptdata = self.vtksgrid3d.GetPointData()
            names = [ptdata.GetArrayName(i) for i in range(ptdata.GetNumberOfArrays())]
            model_names = [self.field_map_3d[k] for k in self.required_keys3d]
            missing = [x for x in model_names if x not in names]
            if len(missing) > 0:
                raise ValueError(f"Missing {missing} array from the input 3D grid")
            # Rename arrays from model-specific names to standard names
            self._apply_field_mapping(self.vtksgrid3d, self.field_map_3d)
        elif suffix == ".vts":
            # Read 3D grid from VTK XML structured grid format
            reader3d = vtk.vtkXMLStructuredGridReader()
            reader3d.SetFileName(self.fname3d)
            reader3d.Update()
            self.vtksgrid3d.ShallowCopy(reader3d.GetOutput())
            # Check for required field arrays (using model-specific names from field_map)
            ptdata = self.vtksgrid3d.GetPointData()
            names = [ptdata.GetArrayName(i) for i in range(ptdata.GetNumberOfArrays())]
            model_names = [self.field_map_3d[k] for k in self.required_keys3d]
            missing = [x for x in model_names if x not in names]
            if len(missing) > 0:
                raise ValueError(f"Missing {missing} array from the input 3D grid")
            # Rename arrays from model-specific names to standard names
            self._apply_field_mapping(self.vtksgrid3d, self.field_map_3d)
        elif suffix == ".npz":
            # Read 3D grid arrays from NumPy .npz file and convert to VTK grid
            npzfile = np.load(self.fname3d)
            reqd = ["x", "y", "z", "vx", "vy", "vz"]
            names = npzfile.files
            missing = [x for x in reqd if x not in names]
            if len(missing) > 0:
                raise ValueError(f"Missing {missing} array(s) from the input 3D grid npz file")

            x = npzfile["x"]
            dims = x.shape
            y = npzfile["y"]
            z = npzfile["z"]
            vx = npzfile["vx"]
            vy = npzfile["vy"]
            vz = npzfile["vz"]

            ll = [x, y, z, vx, vy, vz]
            if not all(a.shape == dims for a in ll):
                raise Exception("input arrays in the 3D grid npz file must all be the same shape")
            if not len(dims) == 3:
                raise Exception("input arrays in the 3D grid npz file must be 3D")

            # ravel the arrays
            x = x.ravel()
            y = y.ravel()
            z = z.ravel()
            vx = vx.ravel()
            vy = vy.ravel()
            vz = vz.ravel()

            # make the coordinates
            ptdata = np.stack([x, y, z]).T
            vptdata = numpy_support.numpy_to_vtk(ptdata)
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(x.size)
            pts.SetData(vptdata)

            # make the grid
            grid = vtk.vtkStructuredGrid()
            grid.SetDimensions(tuple(np.flip(dims)))
            grid.SetPoints(pts)

            # combine the velocity components and add to the grid using standard name
            vel = np.stack([vx, vy, vz]).T
            vtkvel = numpy_support.numpy_to_vtk(vel)
            vtkvel.SetName("velocity")  # standard name
            grid.GetPointData().AddArray(vtkvel)

            # save to the class variable
            self.vtksgrid3d = grid
        else:
            raise TypeError(
                f"{pathlib.Path(self.fname3d).suffix} file type not supported for input 3D grid; "
                f"expected .vtk, .vts, or .npz"
            )

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
        """tuple(str): standard array names that will be present in the 2D grid.

        Returns all standard fields. This includes required fields from field_map_2d
        plus optional fields (like wet_dry) that are either mapped or computed.
        """
        # Always include all standard fields since optional ones are computed if not mapped
        return STANDARD_FIELDS_2D

    @property
    def required_keys3d(self):
        """tuple(str): standard array names required in the input 3D grid."""
        return STANDARD_FIELDS_3D

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

    def write_hdf5(self, obj, name, data):
        """Write cell arrays to HDF5 object.

        Args:
            obj (h5py object): opened HDF5 file to write data to
            name (str): key of the data to be written
            data (NumPy ndarray): data to be written
        """
        obj[name][...] = data

    def write_hdf5_xmf(self, filexmf, time, dims, names, attrnames, dtypes=None, center="Cell"):
        """Body for cell-centered time series data.

        Args:
            filexmf (file): open file to write
            time (float): current simulation time
            dims (tuple): integer values describing dimensions of the grid
            names (list of str): paths to datasets from the root directory in the HDF5 file
            attrnames (list of str): descriptive names corresponding to names
            dtypes (list of str): data types corresponding to names (either Float or Int)
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
            for i, j, k in zip(names, attrnames, dtypes):  # noqa: B905
                self.write_hdf5_xmf_attr(filexmf, dims, i, j, center, k)
        else:
            for i, j in zip(names, attrnames):  # noqa: B905
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
        """ndarray: the inflow/outflow 2D boundary cells.

        A NumPy ndarray with ndims=1 that holds the indices of the upstream and downstream mesh boundary cells.
        As currently implemented, upstream is the i=0 row and downstream is the i=nsc-1 row.
        """
        return self._boundarycells

    @boundarycells.setter
    def boundarycells(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("boundarycells.setter: wrong type, must be NumPy ndarray")
        if values.ndim != 1:
            raise ValueError("boundarycells.setter: ndims must equal 1 for use in np.sortedsearch()")
        if not is_sorted(values):
            print("RiverGrid: boundarycells.setter array must be sorted to use binary search; sorting in-place")
            values.sort()
        self._boundarycells = values

    @property
    def fname2d(self):
        """str: the filename of the 2d grid input."""
        return self._fname2d

    @fname2d.setter
    def fname2d(self, values):
        # Check that input file exists
        inputfile = pathlib.Path(values)
        if not inputfile.exists():
            raise Exception(f"Cannot find 2D input file {inputfile}")
        self._fname2d = values

    @property
    def fname3d(self):
        """str: the filename of the 3d grid input."""
        return self._fname3d

    @fname3d.setter
    def fname3d(self, values):
        # Check that input file exists
        inputfile = pathlib.Path(values)
        if not inputfile.exists():
            raise Exception(f"Cannot find 3D input file {inputfile}")
        self._fname3d = values

    @property
    def nn(self):
        """int: the number of stream-normal points that define the grids.

        For file writing reasons, nn must be >= 1.
        """
        return self._nn

    @nn.setter
    def nn(self, values):
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nn.setter must be int")
        # for file writing reasons, must be >= 1
        values = max(values, 1)
        self._nn = values

    @property
    def nnc(self):
        """int: the number of stream-normal cells defined by the grids.

        For file writing reasons, nnc must be >= 1.
        """
        return self._nnc

    @nnc.setter
    def nnc(self, values):
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nnc.setter must be int")
        # for file writing reasons, must be >= 1
        values = max(values, 1)
        self._nnc = values

    @property
    def ns(self):
        """int: the number of stream-wise points that define the grids.

        For file writing reasons, ns must be >= 1.
        """
        return self._ns

    @ns.setter
    def ns(self, values):
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("ns.setter must be int")
        # for file writing reasons, must be >= 1
        values = max(values, 1)
        self._ns = values

    @property
    def nsc(self):
        """int: the number of stream-wise cells defined by the grids.

        For file writing reasons, nsc must be >= 1.
        """
        return self._nsc

    @nsc.setter
    def nsc(self, values):
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nsc.setter must be int")
        # for file writing reasons, must be >= 1
        values = max(values, 1)
        self._nsc = values

    @property
    def nz(self):
        """int: the number of vertical points that define the grids.

        For file writing reasons, nz must be >= 1.
        """
        return self._nz

    @nz.setter
    def nz(self, values):
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nz.setter must be int")
        # for file writing reasons, must be >= 1
        values = max(values, 1)
        self._nz = values

    @property
    def nzc(self):
        """int: the number of vertical cells defined by the grids.

        For file writing reasons, nzc must be >= 1.
        """
        return self._nzc

    @nzc.setter
    def nzc(self, values):
        # must be basic Python integer type
        if not isinstance(values, int):
            raise TypeError("nzc.setter must be int")
        # for file writing reasons, must be >= 1
        values = max(values, 1)
        self._nzc = values

    @property
    def track3d(self):
        """int: flag that indicates the dimension of the simulation.

        If 3D simulation, track3d=1. Else track3d=0.
        """
        return self._track3d

    @track3d.setter
    def track3d(self, values):
        if not isinstance(values, int):
            raise TypeError("track3d.setter must be int")
        if values < 0 or values > 1:
            raise ValueError("track3d.setter must be 0 or 1")
        self._track3d = values


# @jit(nopython=True)
def is_sorted(arr):
    """An efficient check that a 1D NumPy array is sorted in increasing order.

    Written to allow the use of Numba j.i.t. compiling, though not currently enabled.

    https://stackoverflow.com/questions/47004506/check-if-a-numpy-array-is-sorted

    Args:
        arr (ndarray): array to check

    Returns:
        bool: True if arr is sorted in increasing order, False otherwise
    """
    return all(arr[i + 1] >= arr[i] for i in range(arr.size - 1))
