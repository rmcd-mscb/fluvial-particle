# Development Environment Setup

This document explains how to set up a development environment for fluvial-particle.

## Philosophy

We use a **hybrid approach** that leverages the strengths of both conda and uv:

- **Conda** → Complex compiled dependencies (VTK, HDF5, NumPy with MKL)
- **uv** → Fast installation of pure Python packages
- **Result** → Best of both worlds: optimized binaries + fast dependency resolution

## Dependencies Split

### From Conda (`environment.yml`)

- `python>=3.10` - Python interpreter
- `pip` - Package installer (required for uv)
- `vtk>=9.3.1` - Visualization Toolkit (C++ library)
- `h5py>=3.7.0` - HDF5 interface (C bindings)
- `numpy>=1.20` - NumPy with Intel MKL optimizations

### From uv (`pyproject.toml`)

- All pure Python packages (ruff, pytest, sphinx, etc.)
- The fluvial-particle package itself

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Or update existing environment
conda env update -f environment.yml --prune
```

### 2. Activate Environment

```bash
conda activate fluvial-particle
```

### 3. Install uv (if not already installed)

```bash
pip install uv
```

### 4. Install Package and Python Dependencies

```bash
# For development (includes dev dependencies like ruff, pytest, etc.)
uv pip install -e .[dev]

# For production use only
uv pip install -e .
```

### 5. Install Pre-commit Hooks (Optional but Recommended)

```bash
pre-commit install
```

## Verification

After setup, verify your environment:

```bash
# Check Python version
python --version  # Should be >= 3.10

# Check key packages
python -c "import vtk; print('VTK:', vtk.vtkVersion.GetVTKVersion())"
python -c "import h5py; print('h5py:', h5py.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import fluvial_particle; print('fluvial-particle:', fluvial_particle.__version__)"

# Check dev tools
ruff --version
pytest --version
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fluvial_particle

# Run specific test file
pytest tests/test_main.py

# Use nox for comprehensive testing
nox
```

## Updating Dependencies

### Update Conda Dependencies

Edit `environment.yml` and run:

```bash
conda env update -f environment.yml --prune
```

### Update Python Dependencies

Edit `pyproject.toml` and run:

```bash
uv pip install -e .[dev] --upgrade
```

## Troubleshooting

### "mamba not found" error

**Solution**: Conda now uses libmamba by default. Remove `mamba` from `environment.yml`.

### "poetry not found" error

**Solution**: We no longer use Poetry. Use `uv pip install -e .[dev]` instead.

### VTK or h5py import errors

**Solution**: These should come from conda, not pip. Recreate your environment:

```bash
conda deactivate
conda env remove -n fluvial-particle
conda env create -f environment.yml
conda activate fluvial-particle
uv pip install -e .[dev]
```

### Package conflicts

**Solution**: The conda environment provides VTK, h5py, and numpy. When you run `uv pip install -e .`, it will see these are already satisfied and won't reinstall them (unless versions conflict).

## CI/CD Environment

GitHub Actions uses the same hybrid approach:

1. `conda-incubator/setup-miniconda@v3` sets up conda
2. Installs from `environment.yml`
3. Runs `uv pip install -e .` for the package
4. Runs `nox` for testing

See `.github/workflows/tests.yml` for details.

## Clean Slate (Nuclear Option)

If everything is broken, start fresh:

```bash
# Remove environment
conda deactivate
conda env remove -n fluvial-particle

# Remove any pip-installed packages in base
pip uninstall fluvial-particle -y

# Recreate from scratch
conda env create -f environment.yml
conda activate fluvial-particle
pip install uv
uv pip install -e .[dev]
pre-commit install
```

## Why This Approach?

1. **Performance**: VTK and NumPy from conda-forge are optimized (MKL, etc.)
2. **Reliability**: Complex C++ dependencies are pre-compiled by conda-forge
3. **Speed**: uv is 10-100x faster than pip for pure Python packages
4. **Modern**: Uses current best practices (PEP 621, uv, hatchling)
5. **Flexible**: Works locally and in CI with minimal configuration
