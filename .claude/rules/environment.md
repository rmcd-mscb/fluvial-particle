# Environment Management

This project uses a **hybrid conda + uv** approach:
- **Conda**: Provides Python environment + compiled C/C++ dependencies (VTK, h5py, numpy)
- **uv**: Fast package installer for pure Python dependencies

## Running Commands

**CRITICAL**: All commands must run in the conda environment:

```bash
conda run -n fluvial-particle <command>

# Examples
conda run -n fluvial-particle pytest
conda run -n fluvial-particle ruff check .
conda run -n fluvial-particle python script.py
```

Or activate the environment first:
```bash
conda activate fluvial-particle
pytest
ruff check .
```

## Initial Setup

```bash
# 1. Create conda environment (provides VTK, h5py, numpy, Python)
conda env create -f environment.yml

# 2. Install Python dependencies with uv into the conda env
conda run -n fluvial-particle uv pip install -e ".[dev]"
```

**Important**: Do NOT use `uv sync` - it creates an isolated `.venv` that can't see conda's packages.

## Why Hybrid?

- **VTK has no reliable PyPI wheel** - conda-forge builds work better
- **h5py/numpy** benefit from conda's MKL optimizations
- **uv is fast** for pure Python dependencies (~10x faster than pip)

Note: h5py/numpy/vtk are in `environment.yml` (for local dev) and in pyproject.toml's optional `[ci]` group (for CI/pure-pip installs). Locally, conda's optimized versions are used. In CI, install with `pip install -e ".[ci,dev]"`.

## Adding Dependencies

### Compiled dependencies (rare):
1. Add to `environment.yml`
2. Add to `pyproject.toml` under `[project.optional-dependencies] ci`
3. Run: `conda env update -f environment.yml --prune`

### Python dependencies (common):
1. Add to `pyproject.toml` under `dependencies` or `[project.optional-dependencies] dev`
2. Run: `conda run -n fluvial-particle uv pip install -e ".[dev]"`

## Updating Dependencies

```bash
# Update Python deps
conda run -n fluvial-particle uv pip install --upgrade -e ".[dev]"

# Update conda deps
conda env update -f environment.yml --prune
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'vtk'"
Either:
- Conda environment not activated - run: `conda activate fluvial-particle`
- Or you used `uv sync` instead of `uv pip install` - delete `.venv` and reinstall

### Import errors after adding dependencies
Run: `conda run -n fluvial-particle uv pip install -e ".[dev]"`

### Stale environment
```bash
conda env remove -n fluvial-particle
conda env create -f environment.yml
conda run -n fluvial-particle uv pip install -e ".[dev]"
```

## Future: pixi Migration

We plan to migrate to [pixi](https://pixi.sh) which handles both conda and PyPI deps in a single tool. See GitHub issue #23.
