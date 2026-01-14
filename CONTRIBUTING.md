# Contributor Guide

Thank you for your interest in improving this project. This project is
open-source under the [CCO
license](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
and welcomes contributions in the form of bug reports, feature requests,
and pull requests.

Here is a list of important resources for contributors:

- [Source Code](https://code.usgs.gov/wma/nhgf/fluvparticle)
- [Documentation](https://fluvial-particle.readthedocs.io/)
- [Issue Tracker](https://code.usgs.gov/wma/nhgf/fluvparticle/-/issues)
- [Code of Conduct](https://code.usgs.gov/wma/nhgf/fluvparticle/-/blob/main/CODE_OF_CONDUCT.md)

## How to report a bug

Report bugs on the [Issue Tracker](https://code.usgs.gov/wma/nhgf/fluvparticle/-/issues).

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or
steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker\_]{.title-ref}.

## How to submit changes

Open a [pull request](https://code.usgs.gov/wma/nhgf/fluvparticle/-/merge_requests)
to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- All tests must pass (`pytest`).
- Code must pass linting and formatting checks (`ruff check .` and `ruff format --check .`).
- Include unit tests for new functionality.
- If your changes add functionality, update the documentation
  accordingly.

Feel free to submit early, though---we can always iterate on this.

To run linting and code formatting checks before committing your change,
you can install pre-commit as a Git hook by running the following
command:

```shell
pre-commit install --install-hooks
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate
your approach.

## Development requirements

To set up your development environment do the following.

### Fork the repository

Code edits must be made on your personal fork of
the [main repository](https://code.usgs.gov/wma/nhgf/fluvparticle/-/tree/main).
Read the [Gitlab forking workflow documentation](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html)
for forking instructions and additional information.

### Clone the forked repository

To remind yourself that you're working on a fork of the main repository,
we suggest using the name _fluvial-particle-fork_ for your local repo:

```shell
git clone git@code.usgs.gov:<user_id>/fluvparticle.git fluvial-particle-fork
```

### Setup development environment

This project uses a **hybrid conda + uv** approach:
- **Conda**: Provides Python environment + compiled C/C++ dependencies (VTK, h5py, numpy)
- **uv**: Fast package installer for pure Python dependencies

```shell
# Create conda environment (provides VTK, h5py, numpy)
conda env create -f environment.yml

# Activate the environment
conda activate fluvial-particle

# Install Python dependencies with uv (including dev deps from pyproject.toml)
uv pip install -e ".[dev]"
```

**Important**: Do NOT use `uv sync` (which reads from `uv.lock` and creates an isolated virtual environment in `.venv`) — it creates an isolated `.venv` that cannot see conda's packages. Always run `uv pip install` inside the activated conda environment instead.

It is important to get [pre-commit](https://pre-commit.com/) enabled on
the project, to ensure that certain standards are always met on a git
commit. With several of these, it might fail if files are changed, but
it will change them, and trying the commit a second time will actually
work.

### Testing

Run the test suite with [pytest](https://docs.pytest.org/en/latest/):

```shell
pytest                          # run all tests
pytest -v                       # verbose output
pytest tests/test_main.py       # specific file
pytest -k "test_name"           # by name pattern
```

### Code Quality

```shell
# Lint code
ruff check .

# Format code
ruff format .

# Type checking
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Dependencies

Dependencies are managed in two places:

- **`environment.yml`**: Compiled C/C++ dependencies (VTK, h5py, numpy) - installed via conda
- **`pyproject.toml`**: Pure Python dependencies and dev tools - installed via uv

The compiled dependencies (vtk, h5py, numpy) are in an optional `[ci]` group in pyproject.toml,
used only for CI environments. Locally, conda provides these packages.

#### Local development:
```shell
uv pip install -e ".[dev]"
```

#### CI/pure-pip environments:
```shell
pip install -e ".[ci,dev]"
```

#### Adding compiled dependencies (rare):
1. Add to both `environment.yml` AND `pyproject.toml` under `[project.optional-dependencies] ci`
2. Run: `conda env update -f environment.yml --prune`

#### Adding Python dependencies (common):
1. Add to `pyproject.toml` under `dependencies` or `[project.optional-dependencies] dev`
2. Run: `uv pip install -e ".[dev]"`

#### Note on pandas
`pandas` is included in the `[dev]` optional dependencies. It is required to run tests that use
`pytest.importorskip("pandas")` and to enable optional DataFrame export functionality, but it is
not required for running the core simulation.

### Version Management

This project uses [bump-my-version](https://github.com/callowayproject/bump-my-version)
to update version numbers across multiple files:
- `pyproject.toml`
- `src/fluvial_particle/__init__.py`
- `.bumpversion.cfg`
- `code.json`
- `meta.yaml`

```shell
# Patch: 0.0.3 -> 0.0.4
bump-my-version bump patch

# Minor: 0.0.4 -> 0.1.0
bump-my-version bump minor

# Major: 0.1.0 -> 1.0.0
bump-my-version bump major
```

**Note**: This automatically commits changes and creates git tags.
