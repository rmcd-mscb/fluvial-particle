# Output Code Refactoring Plan

Status: **Partially Implemented**

## Completed (PR #19)

- VTP/PVD output implemented in `src/fluvial_particle/io/`
- `VTPWriter` class for particle VTP output
- `PVDWriter` class for time series collections
- Enable with `output_vtp = True` in settings

## Current Architecture

The output system uses **HDF5 + XDMF** (not native VTK formats):

| Component | Location | Format |
|-----------|----------|--------|
| Particles | `Particles.py:133-265` | HDF5 (`particles.h5`) |
| Particle viz | `Particles.py:267-928` | XDMF (`particles.xmf`) |
| Cells | `RiverGrid.py:281-380` | HDF5 (`cells.h5`) + XDMF |
| MPI support | `create_hdf5()` | h5py with `driver="mpio"` |

## Issues Identified

1. **Hand-written XDMF XML** - Error-prone string formatting
2. **Postprocessing is serial-only** - `RiverGrid.py:282`
3. **No native VTK output** - Missing VTP for particles, VTS for grids
4. **Tight coupling** - I/O logic embedded in physics classes
5. **Double-slash path concatenation** - `f"{output_directory}//particles.h5"`
6. **Inconsistent file context management** - Mixed `with` and manual `.close()`

## Recommended Changes

### Priority 1: Extract I/O Module

Create `src/fluvial_particle/io/` with:
- `__init__.py`
- `hdf5_writer.py` - Existing HDF5 + XDMF logic
- `vtk_writer.py` - New VTK output (VTP, VTS)
- `pvd_writer.py` - PVD time series collections
- `xdmf_builder.py` - Clean XDMF generation

### Priority 2: Add Native VTP Output

```python
class ParticleWriter:
    """Write particle data to VTK PolyData (.vtp)."""

    def write_vtp(self, filepath: Path, particles, time: float) -> None:
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(particles.nparts)
        for i in range(particles.nparts):
            points.SetPoint(i, particles.x[i], particles.y[i], particles.z[i])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # Add vertex cells
        verts = vtk.vtkCellArray()
        for i in range(particles.nparts):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        polydata.SetVerts(verts)

        # Add attributes
        self._add_scalar_array(polydata, "Depth", particles.depth)
        self._add_vector_array(polydata, "Velocity",
                               particles.velx, particles.vely, particles.velz)

        # Add time metadata
        time_arr = vtk.vtkDoubleArray()
        time_arr.SetName("TimeValue")
        time_arr.SetNumberOfTuples(1)
        time_arr.SetValue(0, time)
        polydata.GetFieldData().AddArray(time_arr)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filepath))
        writer.SetInputData(polydata)
        writer.Write()
```

### Priority 3: PVD Collection for Time Series

```python
class PVDWriter:
    """Write PVD collection file for time-varying data."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.entries = []

    def add_timestep(self, time: float, data_file: str) -> None:
        self.entries.append((time, data_file))

    def write(self) -> None:
        with self.filepath.open("w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="1.0">\n')
            f.write('  <Collection>\n')
            for time, datafile in self.entries:
                f.write(f'    <DataSet timestep="{time}" file="{datafile}"/>\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
```

### Priority 4: Parallel VTK Output (PVTP)

```python
class ParallelParticleWriter:
    """Write parallel VTK PolyData (.pvtp + .vtp pieces)."""

    def write_pvtp(self, output_dir: Path, basename: str,
                   particles, time: float, rank: int, size: int) -> None:
        # Each rank writes its own piece
        piece_file = output_dir / f"{basename}_{rank}.vtp"
        self.write_vtp(piece_file, particles, time)

        # Rank 0 writes the parallel collection file
        if rank == 0:
            pvtp_file = output_dir / f"{basename}.pvtp"
            self._write_pvtp_header(pvtp_file, basename, size)
```

### Priority 5: Unified Output Manager

```python
class SimulationOutput:
    """Unified output manager supporting multiple formats."""

    def __init__(self, output_dir: Path, formats: list[str] = None):
        self.output_dir = output_dir
        self.formats = formats or ["hdf5", "vtp"]
        self.writers = self._init_writers()

    def write_particles(self, particles, time, tidx, comm=None):
        for fmt, writer in self.writers.items():
            writer.write(particles, time, tidx, comm)

    def finalize(self):
        for writer in self.writers.values():
            writer.finalize()
```

## Output Format Comparison

| Format | Parallel | Compression | ParaView | Streaming | Best For |
|--------|----------|-------------|----------|-----------|----------|
| HDF5+XDMF | Yes (mpio) | Yes | Yes | Yes | Large datasets, checkpoint/restart |
| VTP+PVD | Via PVTP | Yes | Native | No | Particle visualization |
| VTS+PVD | Via PVTS | Yes | Native | No | Structured grid output |

## Implementation Order

1. Create `fluvial_particle/io/` module structure
2. Move existing HDF5/XDMF code to `hdf5_writer.py`
3. Implement `ParticleWriter` with VTP support
4. Implement `PVDWriter` for time series
5. Add `ParallelParticleWriter` for MPI runs
6. Create `SimulationOutput` unified interface
7. Update `simulation.py` to use new I/O module
8. Add output format selection to user options
