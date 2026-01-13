# Claude Code Instructions for fluvial-particle

## Recent Changes

### PR #15: Field Name Mapping (branch: `add-field-mapping`)
**Goal**: Enable fluvial-particle to work with output from different hydrodynamic models.

#### Completed
- ✅ Added `field_map_2d` and `field_map_3d` required parameters
- ✅ Standard internal field names: `bed_elevation`, `wet_dry`, `shear_stress`, `velocity`, `water_surface_elevation`
- ✅ Users map model-specific names (e.g., Delft-FM's "IBC" → standard "wet_dry")
- ✅ Updated all test configuration files with explicit mappings
- ✅ Updated documentation in `docs/optionsfile.rst` and `docs/example.rst`

#### Key Files
- `src/fluvial_particle/RiverGrid.py` - Field mapping applied during grid reading
- `src/fluvial_particle/Settings.py` - New required parameters
- `docs/optionsfile.rst` - Field mapping documentation

### PR #13: VTS Support (merged to main)
**Goal**: Support modern VTK XML formats and time-dependent flow fields.

#### Completed
- ✅ Added `.vts` (VTK XML Structured Grid) support to `RiverGrid.py`
- ✅ Created VTS test files with TimeValue/TimeStep metadata
- ✅ Added test case: `"simulate with vts input meshes"`
- ✅ Created VTK→VTS conversion notebook: `notebooks/sandbox/convert_vtk_to_vts.qmd`

### Future Work
- 🔲 Time-dependent grid switching for particle tracking
  - Need to efficiently switch between flow field snapshots during simulation
  - Use `TimeValue`/`TimeStep` metadata to map particle time → flow field file
- 🔲 Consider PVD collection files for managing time series
- 🔲 Test with actual time-varying simulation data

---

## Environment Management

This project uses **uv** for dependency and environment management.

### Architecture
- **uv** (`pyproject.toml`): Manages all dependencies and the virtual environment
  - Creates and manages a `.venv` virtual environment
  - Handles all Python dependencies including compiled packages (VTK, h5py, numpy)
  - Development tools (ruff, pytest, mypy, etc.)
  - Documentation tools (Sphinx, myst-parser, etc.)

### Virtual Environment
uv manages a virtual environment in `.venv/` at the project root.

## Running Commands

**CRITICAL**: All commands should be run using the uv-managed environment. Use `uv run` to execute commands:

```bash
# Run any command in the uv environment
uv run <command>

# Examples
uv run pytest
uv run ruff check .
uv run python script.py
```

### Examples
```bash
# Good - runs in uv environment (always up to date)
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src/

# Alternative - activate the venv first
source .venv/bin/activate
pytest
ruff check .
```

## Dependency Management Workflow

### Initial Setup (one-time)
```bash
# Sync all dependencies (creates .venv if needed)
uv sync --all-extras
```

### Adding New Dependencies

#### For runtime dependencies:
1. Add to `pyproject.toml` under `dependencies`
2. Sync with uv:
   ```bash
   uv sync --all-extras
   ```

#### For development dependencies:
1. Add to `pyproject.toml` under `[project.optional-dependencies]`
2. Sync with uv:
   ```bash
   uv sync --all-extras
   ```

### Updating Dependencies

```bash
# Update all dependencies
uv sync --upgrade --all-extras

# Update a specific package
uv add --upgrade <package-name>
```

## Development Workflow

### Running Tests
```bash
uv run pytest
uv run pytest -v  # verbose
uv run pytest tests/test_main.py  # specific file
```

### Code Quality Checks
```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/

# Pre-commit hooks
uv run pre-commit run --all-files
```

### Building Documentation

```bash
# Build HTML documentation locally
uv run sphinx-build docs docs/_build/html

# Build with auto-rebuild on file changes (development mode)
uv run sphinx-autobuild docs docs/_build/html

# Clean previous build
rm -rf docs/_build
```

### ReadTheDocs Setup

The project is configured to automatically build documentation on ReadTheDocs from this GitHub repository.

**Configuration files:**
- `.readthedocs.yml` - ReadTheDocs build configuration
- `docs/conf.py` - Sphinx configuration
- `docs/requirements.txt` - Documentation dependencies

**Setting up a new ReadTheDocs project:**

1. Go to https://readthedocs.org/
2. Log in with your GitHub account
3. Click "Import a Project"
4. Select `rmcd-mscb/fluvial-particle` from your GitHub repositories
5. Configure project settings:
   - **Name**: fluvial-particle
   - **Repository URL**: https://github.com/rmcd-mscb/fluvial-particle
   - **Default branch**: main
6. The build will automatically use `.readthedocs.yml` configuration
7. Enable "Build pull requests" in Admin → Advanced Settings for PR previews

**Manual build trigger:**
- Go to your ReadTheDocs project dashboard
- Click "Build Version" to manually trigger a documentation build

**Documentation URL:**
- https://fluvial-particle.readthedocs.io/en/latest/

### Version Bumping

The project uses `bump-my-version` to manage version numbers across multiple files:
- `pyproject.toml` - Project metadata
- `src/fluvial_particle/__init__.py` - Python package version
- `.bumpversion.cfg` - Legacy config file
- `code.json` - USGS code metadata
- `meta.yaml` - Conda package metadata

```bash
# Patch version (0.0.3 -> 0.0.4)
uv run bump-my-version bump patch

# Minor version (0.0.4 -> 0.1.0)
uv run bump-my-version bump minor

# Major version (0.1.0 -> 1.0.0)
uv run bump-my-version bump major
```

**Note**: bump-my-version automatically commits changes and creates git tags.

## Why uv?

1. **Fast**: uv is significantly faster than pip or conda for dependency resolution and installation
2. **Reproducible**: `uv.lock` ensures exact reproducibility across environments
3. **Simple**: Single tool manages both dependencies and virtual environment
4. **Modern**: Handles compiled packages (VTK, h5py, numpy) without issues

## Important Notes

- **Always use `uv run`**: This ensures commands run in the correct environment with up-to-date dependencies
- **Sync after changes**: After modifying `pyproject.toml`, always run `uv sync --all-extras`
- **Don't mix pip and uv**: Use uv for all Python package management
- **Lock file**: The `uv.lock` file should be committed to version control for reproducibility

## Troubleshooting

### "Command not found" errors
Make sure you're using `uv run <command>` or have activated the virtual environment.

### Import errors after adding dependencies
Run `uv sync --all-extras` to install new dependencies.

### Stale environment
If you encounter strange behavior, try:
```bash
rm -rf .venv
uv sync --all-extras
```

## Editing Jupyter Notebooks (.ipynb)

**Issue**: `.ipynb` files are JSON files, but VS Code may present them in an internal XML representation. Standard text editing tools (`replace_string_in_file`) may appear to work but changes don't persist.

**Solution**: Use one of these approaches:

### Option 1: Use `edit_notebook_file` tool (Preferred for VS Code)
```python
edit_notebook_file(
    cellId="#VSC-xxxxx",  # Cell ID from the notebook
    editType="edit",
    filePath="/path/to/notebook.ipynb",
    language="python",
    newCode="# your updated code here"
)
```

### Option 2: Edit the raw JSON directly (Most reliable)
When the notebook editing tools don't persist changes, use terminal commands to edit the raw JSON:

```bash
# Example: Replace os.path with pathlib.Path
sed -i 's/os\.path\.exists(/Path(/g' notebook.ipynb
sed -i 's/os\.path\.getsize(/Path(/g' notebook.ipynb
```

**Key takeaway**: If notebook edits aren't persisting, close the notebook in VS Code and edit the raw JSON file directly with `sed`, `awk`, or standard text replacement tools.

## Supported Input File Formats

The project supports multiple grid file formats for 2D and 3D input meshes:

| Format | Extension | Description |
|--------|-----------|-------------|
| VTK XML Structured Grid | `.vts` | **Recommended** - Modern binary format with compression |
| VTK Legacy | `.vtk` | ASCII or binary, widely compatible |
| NumPy Archive | `.npz` | Python-specific compressed format |

### VTS File Time Metadata

VTS files can include time metadata in their FieldData section:
- `TimeValue` (Float64): Time in seconds for ParaView animation support
- `TimeStep` (Int32): Integer timestep index for efficient file lookup

### Field Name Mapping

Different hydrodynamic models use different array names. The `field_map_2d` and `field_map_3d` parameters map model-specific names to standard internal names:

**Standard 2D fields:**
- `bed_elevation` - bed/bottom elevation
- `wet_dry` - wet/dry indicator (1=wet, 0=dry)
- `shear_stress` - shear stress magnitude
- `velocity` - velocity vector
- `water_surface_elevation` - water surface elevation

**Standard 3D fields:**
- `velocity` - velocity vector

**Example for Delft-FM output:**
```python
field_map_2d = {
    "bed_elevation": "Elevation",
    "wet_dry": "IBC",
    "shear_stress": "ShearStress (magnitude)",
    "velocity": "Velocity",
    "water_surface_elevation": "WaterSurfaceElevation",
}
field_map_3d = {
    "velocity": "Velocity",
}
```

Note: For `.npz` files, these mappings are not used as the npz format has its own internal naming convention.
