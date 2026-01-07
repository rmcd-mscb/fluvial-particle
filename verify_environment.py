#!/usr/bin/env python
"""Verify the fluvial-particle development environment is set up correctly."""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need >= 3.10)")
    return False


def check_conda_packages():
    """Check conda-installed packages."""
    print("\nChecking conda packages...")
    packages = {
        "vtk": "Visualization Toolkit",
        "h5py": "HDF5 interface",
        "numpy": "NumPy",
    }

    all_ok = True
    for pkg_name, description in packages.items():
        try:
            module = __import__(pkg_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {pkg_name} {version} - {description}")
        except ImportError:
            print(f"  ✗ {pkg_name} - {description} NOT FOUND")
            all_ok = False

    return all_ok


def check_uv_packages():
    """Check uv/pip-installed packages."""
    print("\nChecking development tools...")
    tools = {
        "ruff": "Linter and formatter",
        "pytest": "Testing framework",
        "nox": "Test automation",
        "pre_commit": "Pre-commit hooks",
    }

    all_ok = True
    for tool_name, description in tools.items():
        try:
            module = __import__(tool_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {tool_name} {version} - {description}")
        except ImportError:
            print(f"  ✗ {tool_name} - {description} NOT FOUND")
            print("     Install with: uv pip install -e .[dev]")
            all_ok = False

    return all_ok


def check_package_installation():
    """Check if fluvial_particle is installed."""
    print("\nChecking fluvial_particle package...")
    try:
        import fluvial_particle

        version = getattr(fluvial_particle, "__version__", "unknown")
        print(f"  ✓ fluvial_particle {version}")

        # Check if it's editable install
        import fluvial_particle as fp

        package_path = Path(fp.__file__).parent
        if (package_path.parent.parent / "pyproject.toml").exists():
            print(f"  ✓ Installed in editable mode from: {package_path.parent.parent}")
        else:
            print(f"  ⚠ Not an editable install (installed from: {package_path})")

        return True
    except ImportError:
        print("  ✗ fluvial_particle NOT FOUND")
        print("     Install with: uv pip install -e .[dev]")
        return False


def check_environment_files():
    """Check if key configuration files exist."""
    print("\nChecking configuration files...")
    files = {
        "environment.yml": "Conda environment specification",
        "pyproject.toml": "Python package configuration",
        ".pre-commit-config.yaml": "Pre-commit hooks configuration",
    }

    all_ok = True
    for filename, description in files.items():
        path = Path(filename)
        if path.exists():
            print(f"  ✓ {filename} - {description}")
        else:
            print(f"  ✗ {filename} - {description} NOT FOUND")
            all_ok = False

    return all_ok


def main():
    """Run all checks."""
    print("=" * 70)
    print("Fluvial-Particle Environment Verification")
    print("=" * 70)

    checks = [
        check_python_version(),
        check_conda_packages(),
        check_uv_packages(),
        check_package_installation(),
        check_environment_files(),
    ]

    print("\n" + "=" * 70)
    if all(checks):
        print("✓ All checks passed! Your environment is correctly set up.")
        print("=" * 70)
        print("\nYou can now:")
        print("  - Run tests: pytest")
        print("  - Run linter: ruff check .")
        print("  - Run formatter: ruff format .")
        print("  - Run all checks: nox")
        print("  - Install pre-commit: pre-commit install")
        return 0
    print("✗ Some checks failed. Please review the errors above.")
    print("=" * 70)
    print("\nTo fix:")
    print("  1. Ensure conda environment is activated: conda activate fluvial-particle")
    print("  2. Install development dependencies: uv pip install -e .[dev]")
    print("  3. Run this script again to verify")
    return 1


if __name__ == "__main__":
    sys.exit(main())
