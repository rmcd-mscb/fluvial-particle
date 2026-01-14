# History

## 0.0.5 (2026-01-14)

### New Features

#### Time-Varying Grid Support
- Added `TimeVaryingGrid` class for unsteady flow simulations with pre-computed velocity fields
- Supports temporal interpolation between grid timesteps: `linear`, `nearest`, and `hold` modes
- Sliding window approach keeps only 2 grids in memory for efficient large simulations
- Velocity blending between timesteps for smooth particle transport

#### VTP/PVD Output Format
- Added optional VTK PolyData (.vtp) output for native ParaView support
- PVD collection files with timestamp information for time-series visualization
- Enable with `output_vtp = True` in settings file

#### Notebook Convenience API
- New `SimulationResults` class for easy access to simulation output
  - `get_positions()`, `get_positions_2d()` for particle coordinates
  - `get_property()`, `get_velocities()`, `get_depths()` for particle data
  - `to_dataframe()` for optional pandas DataFrame export
  - `summary()` for quick overview of results
- New `run_simulation()` function wrapping verbose setup into single call
- Context manager support for automatic HDF5 file cleanup

#### CLI Improvements
- Added comprehensive `--help` with program description and usage examples
- Added `--version` flag
- Added `--init` flag to generate template settings file (`user_options.py`)

#### Multi-Model Support
- Added `field_map_2d` and `field_map_3d` for mapping standard field names to model-specific names
- Supports output from Delft-FM, iRIC, HEC-RAS, and other hydrodynamic models
- Optional `wet_dry` field - auto-computed from depth when not provided

#### VTS File Format Support
- Added VTK XML Structured Grid (.vts) format as recommended input format
- Binary format with compression, 5-10x smaller than legacy VTK

### Documentation
- Added time-varying grid settings documentation
- Added Copilot code review instructions (`.github/copilot-instructions.md`)
- Updated options file reference with all new parameters

### Bug Fixes
- Fixed velocity blending for time-varying grids (interpolation weight was calculated but not applied)

## 0.0.4 (2026-01-07)

### Bug Fixes
- Fixed VTK 9.3+ API compatibility issues with deprecated `GetPoints()` method
- Replaced deprecated `GetPoints()` with `GetPointData()` in RiverGrid class

### Tooling & Infrastructure
- **Dependency Management**: Migrated from Poetry to uv for faster, more reliable dependency resolution
- **Code Quality**: Consolidated 11+ tools (black, flake8, isort, darglint, etc.) into Ruff for unified linting and formatting
- **Pre-commit**: Modernized pre-commit hooks to use Ruff
- **Security**: Updated safety check to use inline ignore pattern for false positives
- **Type Checking**: Maintained mypy configuration with strict typing enforcement
- **Testing**: Retained pytest and coverage infrastructure
- **Documentation**: Kept Sphinx documentation tooling (ReadTheDocs compatible)

### Dependencies
- Updated h5py to version 3.13.0
- Updated vtk to version 9.4.0
- Updated mpi4py to version 4.0.2
- Updated numpy to version 2.2.2
- Added bump-my-version for version management

### Development Experience
- Significantly improved development workflow with faster dependency installation
- Reduced configuration complexity by eliminating redundant tool configurations
- Maintained strict code quality standards with modernized tooling

## 0.0.3

Previous releases (details not available in current history)

## 0.0.1-dev0 (2021-08-17)

- Initial development release
