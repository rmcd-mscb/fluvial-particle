# Copilot Code Review Instructions for fluvial-particle

## Project Overview

**fluvial-particle** is a Lagrangian particle tracking library for simulating particle transport in rivers. It uses velocity fields from hydrodynamic models (Delft-FM, iRIC, HEC-RAS) to track particles through fluvial environments.

### Key Technologies
- **VTK (Visualization Toolkit)**: Core dependency for mesh handling and interpolation
- **HDF5 (h5py)**: Output format for simulation results
- **NumPy**: Numerical computations
- **MPI (mpi4py)**: Optional parallel execution support

## Architecture

### Core Components

1. **Particles.py** (`src/fluvial_particle/Particles.py`)
   - Base `Particles` class and subclasses (`LarvalTopParticles`, `LarvalBotParticles`, `FallingParticles`)
   - Handles particle state, movement, and physics
   - Uses VTK probe filters for field interpolation

2. **RiverGrid.py** (`src/fluvial_particle/RiverGrid.py`)
   - Manages 2D/3D mesh data from hydrodynamic models
   - Handles multiple file formats: `.vts` (VTK XML), `.vtk` (legacy), `.npz` (NumPy)
   - Field name mapping system for multi-model support

3. **TimeVaryingGrid** (`src/fluvial_particle/grids/time_varying_grid.py`)
   - Wrapper for time-dependent flow fields
   - Sliding window approach (2 grids in memory)
   - Temporal interpolation: linear, nearest, hold

4. **Settings.py** (`src/fluvial_particle/Settings.py`)
   - Configuration management (dict subclass)
   - Reads user options files (Python scripts)

5. **simulation.py** (`src/fluvial_particle/simulation.py`)
   - Main simulation loop
   - Orchestrates particles, grids, and I/O

## Code Review Checklist

### Python Style
- [ ] Code passes `ruff check` (linting) and `ruff format` (formatting)
- [ ] Type hints used for function signatures (Python 3.10+ style: `list[int]` not `List[int]`)
- [ ] Docstrings follow NumPy style for public functions
- [ ] No unused imports (F401) except where `# noqa: F401` is appropriate
- [ ] Explicit `encoding="utf-8"` on all file operations (PLW1514)

### Testing
- [ ] New features have corresponding tests in `tests/`
- [ ] Tests use `pytest` fixtures and `numpy.testing` assertions
- [ ] Integration tests use `TemporaryDirectory` for output
- [ ] Test data files go in `tests/data/`

### Architecture Patterns
- [ ] Field access uses `field_map_2d`/`field_map_3d` mappings, not hardcoded names
- [ ] VTK operations properly clean up (though Python GC usually handles this)
- [ ] MPI code checks `comm.Get_rank() == 0` for print statements
- [ ] Grid loading validates required fields exist

### Common Issues to Watch For

1. **VTK Array Handling**
   ```python
   # WRONG: VTK arrays share memory, modifications affect original
   arr = numpy_support.vtk_to_numpy(vtk_array)
   arr *= 2  # Modifies VTK data!

   # CORRECT: Copy if you need to modify
   arr = numpy_support.vtk_to_numpy(vtk_array).copy()
   ```

2. **Time-Varying Grid Interpolation**
   - When `get_interpolation_weight() > 0`, both current AND next grids must be queried
   - Velocity blending: `(1 - weight) * current + weight * next`

3. **Particle Bounds**
   - Particles must stay within `[vertbound, 1-vertbound]` of water column
   - Check `out_of_grid()` before interpolating fields

4. **HDF5 Parallel I/O**
   - Use `driver="mpio"` with MPI communicator for parallel writes
   - Collective operations required for dataset creation

### Performance Considerations
- [ ] Avoid repeated VTK pipeline rebuilds (expensive)
- [ ] Use vectorized NumPy operations, not Python loops
- [ ] TimeVaryingGrid keeps only 2 grids in memory (sliding window)
- [ ] Large particle counts should use MPI parallelization

### Security
- [ ] User options files are executed as Python - document this clearly
- [ ] No hardcoded paths that assume specific directory structures
- [ ] Validate file paths exist before operations

## Environment Notes

This project uses a **hybrid conda + uv** approach:
- **Conda**: VTK, h5py, numpy (compiled C++ dependencies)
- **uv**: Pure Python dependencies

CI runs with PyPI packages only (no conda), so compiled deps must be in `pyproject.toml`.

## File Format Support

| Format | Extension | Use Case |
|--------|-----------|----------|
| VTK XML Structured Grid | `.vts` | Recommended, binary with compression |
| VTK Legacy | `.vtk` | ASCII/binary, older tool compatibility |
| NumPy Archive | `.npz` | Python-specific workflows |

## Required Fields

### 2D Mesh (`field_map_2d`)
- `bed_elevation` (required)
- `shear_stress` (required)
- `velocity` (required)
- `water_surface_elevation` (required)
- `wet_dry` (optional - computed from depth if omitted)

### 3D Mesh (`field_map_3d`)
- `velocity` (required)

## Links

- Documentation: https://fluvial-particle.readthedocs.io/
- Options file reference: https://fluvial-particle.readthedocs.io/en/latest/optionsfile.html
