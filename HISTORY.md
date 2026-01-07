# History

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
