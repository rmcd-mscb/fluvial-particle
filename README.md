# Fluvial Particle

[![PyPI](https://img.shields.io/pypi/v/fluvial-particle.svg)](https://pypi.org/project/fluvial-particle/)
[![Status](https://img.shields.io/pypi/status/fluvial-particle.svg)](https://pypi.org/project/fluvial-particle/)
[![Python Version](https://img.shields.io/pypi/pyversions/fluvial-particle)](https://pypi.org/project/fluvial-particle/)
[![License](https://img.shields.io/pypi/l/fluvial-particle)](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
[![Read the Docs](https://img.shields.io/readthedocs/fluvial-particle/latest.svg?label=Read%20the%20Docs)](https://fluvial-particle.readthedocs.io/)

[![Tests]](https://code.usgs.gov/%{project_path}/badges/%{default_branch}/pipeline.svg)
[![Codecov](https://codecov.io/gh/rmcd-mscb/fluvial-particle/branch/main/graph/badge.svg)](https://codecov.io/gh/rmcd-mscb/fluvial-particle)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- TODO

## Requirements

per Cliff Hills setup description: <https://code.usgs.gov/wma/nhgf/pygeoapi-plugin-cookiecutter>

To set up your development environment do the following.

- fork the reposistory

```{.sourceCode .console}
# to remind yourself your working on a fork
git clone git@code.usgs.gov:<user_id>/fluvparticle.git fluvial-partile-fork
```

Setup Development Environment

```{.sourceCode .console}
conda env create -f environment.yml
conda develop -n {{cookiecutter.project_name}} src
conda activate {{cookiecutter.project_name}}
pip install -r requirements.dev
```

It is important to get [preccommit](https://pre-commit.com/) enabled on
the project, to ensure that certain standards are always met on a git
commit. With several of these, it might fail if files are changed, but
it will change them, and trying the commit a second time will actually
work.

### Git hook configuration

```{.sourceCode .console}
pre-commit install --install-hooks
```

### Testing

[Nox](https://nox.thea.codes/) is used for testing everything, with
several sessions built-in. To run the full suite of tests, simply use:

```{.sourceCode .console}
nox
```

The different sessions are:

- `pre-commit` -- validates that the
  [preccommit](https://pre-commit.com/) checks all come back clean.
- `safety` -- validates the [Safety](https://github.com/pyupio/safety)
  of all production dependencies.
- `mypy` -- validates the type-hints for the application using
  [mypy](http://mypy-lang.org/).
- `tests` -- runs all [pytest](https://docs.pytest.org/en/latest/)
  tests.
- `typeguard` -- runs all [pytest](https://docs.pytest.org/en/latest/)
  tests, validates with
  [Typeguard](https://github.com/agronholm/typeguard).
- `xdoctest` -- runs any and all documentation examples with
  [xdoctest](https://github.com/Erotemic/xdoctest).
- `docs-build` -- builds the full set of generated API docs with
  [Sphinx](http://www.sphinx-doc.org/).

These can be run individually with the following command:

```{.sourceCode .console}
nox -s <session>
```

Replace `<session>` with the name of the session give above, i.e.:

```{.sourceCode .console}
nox -s mypy
```

You can also simply run [pytest](https://docs.pytest.org/en/latest/)
tests, by using the command:

```{.sourceCode .console}
pytest tests
```

### Dependencies

Production dependencies are duplicated, in both `requirements.txt` and
`environment.yml` due to how [conda](https://www.anaconda.com) does not
work with the `requirements.txt` file. It is necessary for both files to
be updated as dependencies are added.

Development dependencies are contained in `requirements.dev`.

### Version Management

The projects made by this cookiecutter use
[Bump2version](https://github.com/c4urself/bump2version) for version
management. The default version that the project starts with is a
developmental version, `0.0.1-dev0`. In github, this should be
auto-incremented on each commit to the next dev build number. To manage
the version changes yourself, you can use the
[Bump2version](https://github.com/c4urself/bump2version) command:

```{.sourceCode .console}
bump2version <part>
```

Where `<part>` is one of:

- `major`
- `minor`
- `patch`
- `build`

Note:
: This makes a `dev` version, which does not write a tag into git. It is just useful for development purposes and not the version that is recommended for a release version. The version string will be formatted as: `<major>.<minor>.<patch>-dev<build>`

To do a production release, use the command:

```{.sourceCode .console}
bump2version --tag release
```

This will add a tag in the git repository noting the version.

Note:
: The version string for this will be: `<major>.<minor>.<patch>`

## Utilities

- Packaging and dependency management with
  [conda](https://www.anaconda.com)
- Test automation with [Nox](https://nox.thea.codes/)
- Linting with [preccommit](https://pre-commit.com/) and
  [Flake8](http://flake8.pycqa.org)
- Continuous integration with [GitHub
  Actions](https://github.com/features/actions) or
  [Travis-CI](https://travis-ci.com)
- Documentation with [Sphinx](http://www.sphinx-doc.org/) and [Read
  the Docs](https://readthedocs.org/)
- Automated uploads to [PyPI](https://pypi.org/) and
  [TestPyPI](https://test.pypi.org/)
- Automated release notes with [Release
  Drafter](https://github.com/release-drafter/release-drafter)
- Automated dependency updates with
  [Dependabot](https://dependabot.com/)
- Code formatting with [Black](https://github.com/psf/black) and
  [Prettier](https://prettier.io/)
- Testing with [pytest](https://docs.pytest.org/en/latest/)
- Code coverage with [Coverageppy](https://coverage.readthedocs.io/)
- Coverage reporting with [Codecov](https://codecov.io/)
- Command-line interface with
  [Click](https://click.palletsprojects.com/)
- Static type-checking with [mypy](http://mypy-lang.org/)
- Runtime type-checking with
  [Typeguard](https://github.com/agronholm/typeguard)
- Security audit with [Bandit](https://github.com/PyCQA/bandit) and
  [Safety](https://github.com/pyupio/safety)
- Check documentation examples with
  [xdoctest](https://github.com/Erotemic/xdoctest)
- Generate API documentation with
  [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
  and
  [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- Generate command-line reference with
  [sphinxcclick](https://sphinx-click.readthedocs.io/)
- Manage project labels with [GitHub
  Labeler](https://github.com/marketplace/actions/github-labeler)
- Manage project versions with
  [Bump2version](https://github.com/c4urself/bump2version)
- Automatic loading/unloading of [conda](https://www.anaconda.com)
  environment with [direnv](https://direnv.net/)

## Installation

You can install _Fluvial_Particle_ via pip from `PyPI_`:

$ pip install fluvial-particle

## Usage

TODO

## Contributing

Contributions are very welcome.
To learn more, see the `Contributor Guide_`\.

## License

Distributed under the terms of the `CCO 1.0 license_`, Fluvial Particle is free and open source software.

## Issues

If you encounter any problems,
please `file an issue_` along with a detailed description.

## Credits

This project was generated from `@hillc-usgs_`'s `Pygeoapi Plugin Cookiecutter_` template.

..\_@hillc-usgs: <https://github.com/hillc-usgs>
..\_Cookiecutter: <https://github.com/audreyr/cookiecutter>
..\_CCO 1.0 license: <https://creativecommons.org/publicdomain/zero/1.0/legalcode>
..\_PyPI: <https://pypi.org/>
..\_Pygeoapi Plugin Cookiecutter: <https://code.usgs.gov/wma/nhgf/pygeoapi-plugin-cookiecutter>
..\_file an issue: <https://github.com/rmcd-mscb/fluvial-particle/issues>
..\_pip: <https://pip.pypa.io/>
.. github-only
..\_Contributor Guide: CONTRIBUTING.rst
..\_Usage: <https://fluvial-particle.readthedocs.io/en/latest/usage.html>
