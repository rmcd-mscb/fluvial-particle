# Claude Code Instructions for fluvial-particle

## Environment Management

This project uses a **hybrid conda + uv** approach for dependency management:

### Architecture
- **Conda** (`environment.yml`): Manages compiled C/C++ dependencies that benefit from conda's binary distribution
  - VTK (complex C++ visualization library)
  - h5py (HDF5 with C bindings)
  - numpy (with MKL optimizations)
  - Python interpreter

- **uv** (`pyproject.toml`): Manages all pure Python dependencies
  - Development tools (ruff, pytest, mypy, etc.)
  - Documentation tools (Sphinx, myst-parser, etc.)
  - All other Python packages

### Conda Environment Name
`fluvial-particle`

## Running Commands

**CRITICAL**: All commands MUST be run within the conda environment. Use one of these approaches:

### Option 1: Activate environment first (for interactive work)
```bash
conda activate fluvial-particle
<your commands here>
```

### Option 2: Use conda run (for single commands)
```bash
conda run -n fluvial-particle <command>
```

### Examples
```bash
# Good - runs in environment
conda run -n fluvial-particle uv sync
conda run -n fluvial-particle pytest
conda run -n fluvial-particle ruff check .

# Bad - runs in base environment (wrong!)
uv sync
pytest
ruff check .
```

## Dependency Management Workflow

### Initial Setup (one-time)
```bash
# 1. Create conda environment with compiled dependencies
conda env create -f environment.yml

# 2. Activate the environment
conda activate fluvial-particle

# 3. Install Python dependencies with uv
uv sync --all-extras
```

### Adding New Dependencies

#### For compiled dependencies (rare):
1. Add to `environment.yml`
2. Update the environment:
   ```bash
   conda env update -f environment.yml --prune
   ```

#### For Python dependencies (common):
1. Add to `pyproject.toml` under `dependencies` or `[project.optional-dependencies]`
2. Sync with uv:
   ```bash
   conda run -n fluvial-particle uv sync --all-extras
   ```

### Updating Dependencies

```bash
# Update all dependencies
conda run -n fluvial-particle uv sync --upgrade --all-extras

# Update conda dependencies
conda env update -f environment.yml --prune
```

## Development Workflow

### Running Tests
```bash
conda run -n fluvial-particle pytest
```

### Code Quality Checks
```bash
# Lint and format
conda run -n fluvial-parameter ruff check .
conda run -n fluvial-particle ruff format .

# Type checking
conda run -n fluvial-particle mypy src/

# Pre-commit hooks
conda run -n fluvial-particle pre-commit run --all-files
```

### Building Documentation
```bash
conda run -n fluvial-particle sphinx-build docs docs/_build
```

### Version Bumping
```bash
# Patch version (0.0.3 -> 0.0.4)
conda run -n fluvial-particle bump-my-version bump patch

# Minor version (0.0.4 -> 0.1.0)
conda run -n fluvial-particle bump-my-version bump minor

# Major version (0.1.0 -> 1.0.0)
conda run -n fluvial-particle bump-my-version bump major
```

## Why This Approach?

1. **Conda handles compiled code well**: VTK and HDF5 have complex C++ dependencies that conda manages better than pip
2. **uv is fast for Python packages**: uv resolves and installs pure Python dependencies much faster than pip or conda
3. **Best of both worlds**: Leverages each tool's strengths
4. **Reproducible**: `environment.yml` locks conda deps, uv uses `uv.lock` for Python deps

## Important Notes

- **Always check your environment**: Before running commands, verify you're in the correct environment
- **Sync after changes**: After modifying `pyproject.toml`, always run `uv sync`
- **Don't mix pip and uv**: Use uv for all Python package management
- **Test in clean environment**: Periodically recreate the environment from scratch to ensure reproducibility

## Troubleshooting

### "Command not found" errors
You're probably in the base environment. Activate `fluvial-particle` first.

### Import errors after adding dependencies
Run `conda run -n fluvial-particle uv sync --all-extras` to install new dependencies.

### Conda and uv conflict
Conda provides the base dependencies (Python, VTK, h5py, numpy). uv installs everything else into the same environment. They don't conflict because uv respects conda-installed packages.
