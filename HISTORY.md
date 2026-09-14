# History

## Unreleased

### Bug Fixes
- Fixed a crash (`ValueError` in `Particles._is_part_wet`) when the last active particles leave the grid in the same time step

### Tests
- Analytical acceptance tests for the 2D/3D solver on a synthetic uniform straight channel (`tests/test_analytical.py`, `write_straight_channel()` in `tests/support.py`): exact advection and exit step, Gaussian plume moments and normality, and well-mixed lateral (dry margins) and vertical (bed/surface) distributions that stay uniform

## 0.1.0 (2026-09-14)

### New Features

#### 1D River-Network Particle Tracking
- New `fluvial_particle.network` subpackage: a passive particle tracker for river networks driven by the network hydraulics NetCDF export from [pywatershed](https://github.com/EC-USGS/pywatershed) (`pywatershed.utils.export_network_hydraulics`): reach topology, daily hydraulics, and optional map polylines
- `FileHydraulicsProvider` validates the schema, units, and topology at open, streams two time slices at a time, interpolates in time (`linear` or `hold`), and subsets reaches by id or by the upstream closure of an outlet
- Sources are mass loadings: slugs, constant or tabulated loading, and concentration curves converted with the reach flow; particles are equal-mass samples of the loading; `estimate_particles()` sizes the particle budget for a target bin occupancy
- Solver: exact advection with time carry across reach boundaries, one Fischer-dispersion kick per step with displacement carry, upstream hops by recorded history then by flow-weighted parent, outlets record exit times
- Output `network_particles.nc` through h5netcdf (serial or MPI-IO); `NetworkResults` post-processing: positions, map positions, sub-reach bins, concentration with optional Gaussian smoothing, arrival times and breakthrough histograms, DataFrame and VTP/PVD export
- `[network]` TOML table (`NetworkConfig`), `run_network_simulation()`, and the `fluvial_particle_network` / `fluvial_particle_network_mpi` entry points
- Analytical acceptance tests on a uniform chain: exact travel times, Gaussian plume moments, inverse-Gaussian arrivals with the documented first-passage bias bound
- Demo notebooks: `notebooks/network-drb-demo.ipynb` (Delaware River Basin) and `notebooks/network-chain-dispersion-demo.ipynb` (uniform chain against the analytical solution)
- User documentation in `docs/network.rst`; design spec and implementation plan under `docs/superpowers/`

#### TOML Configuration
- Settings files may be TOML (recommended) or Python; `get_settings_template()` returns the template for notebooks (#37, #39)
- bump-my-version configuration migrated from `.bumpversion.cfg` to `pyproject.toml`

### Bug Fixes
- `inspect_grid()` shows the 2D velocity field even when `Track3D = 1`

### Infrastructure
- CI: pin `vtk<9.7` because VTK 9.7 changes `vtkProbeFilter` results for the 2D/3D solver (#41); pin the nox ruff version to the pre-commit hook's; replace `safety check` with pip-audit; declare `linkify-it-py` for the documentation build
- Release workflow rewritten: on a version tag it builds with hatchling, publishes the GitHub release notes with the distributions attached, and uploads to PyPI only when a `PYPI_TOKEN` secret is configured; Labeler permissions fixed
- Repository links point at GitHub (`github.com/rmcd-mscb/fluvial-particle`)

### Dependencies
- New runtime dependencies: `xarray`, `h5netcdf`
- Dev extras: `scipy`, `matplotlib`, `pip-audit` (replaces `safety`)

## 0.0.6 (2026-01-15)

### New Features

#### Multiple Shear Velocity (u*) Computation Methods
- Support for 7 different methods to compute shear velocity:
  - Direct `ustar` field mapping
  - `shear_stress` field (τ_b) - existing default
  - `manning_n` - Manning's roughness coefficient (scalar or field)
  - `chezy_c` - Chézy coefficient (scalar or field)
  - `darcy_f` - Darcy-Weisbach friction factor (scalar or field)
  - `energy_slope` field
  - `tke` field (turbulent kinetic energy for RANS models)
- Automatic method detection with priority-based selection
- Optional `ustar_method` setting to force specific method
- Configurable `water_density` for shear stress conversion (default: 1000 kg/m³)

#### Grid Inspection API
- New `inspect_grid()` function for exploring grid data before simulation
- Returns comprehensive dict with grid dimensions, bounds, field statistics
- Displays hydraulic summary (depth, velocity, shear stress ranges)
- Shows detected u* computation method and available alternatives
- Supports time-varying grids with timestep selection

#### PyVista Visualization Helpers
- New `SimulationResults.to_pyvista()` - convert particle positions to PyVista PolyData
- New `SimulationResults.trajectories_to_pyvista()` - create polylines for particle paths
- New `SimulationResults.to_pyvista_sequence()` - get all timesteps for animations
- All methods use lazy imports (pyvista is optional dependency)

### Infrastructure
- Fixed hybrid conda + uv dependency management workflow
- Updated environment.yml and pyproject.toml for better local development

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
