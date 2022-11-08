# Welcome to *fluvial-particle*

[![PyPI](https://img.shields.io/pypi/v/fluvial-particle.svg)](https://pypi.org/project/fluvial-particle/)
[![Status](https://img.shields.io/pypi/status/fluvial-particle.svg)](https://pypi.org/project/fluvial-particle/)
[![Python Version](https://img.shields.io/pypi/pyversions/fluvial-particle)](https://pypi.org/project/fluvial-particle/)
[![License](https://img.shields.io/pypi/l/fluvial-particle)](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
[![Read the Docs](https://img.shields.io/readthedocs/fluvial-particle/latest.svg?label=Read%20the%20Docs)](https://fluvial-particle.readthedocs.io/)

[![Tests](https://code.usgs.gov/wma/nhgf/fluvparticle/badges/main/pipeline.svg)](https://code.usgs.gov/wma/nhgf/fluvparticle/-/commits/main)
[![Codecov](https://codecov.io/gh/rmcd-mscb/fluvial-particle/branch/main/graph/badge.svg)](https://codecov.io/gh/rmcd-mscb/fluvial-particle)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package to efficiently model active- and passive-particle transport in flowing rivers.

![An animated image shows a fluvial-particle simulation output in the Kootenai River, Idaho, USA](./data/kootenai_2to27_particles_fpc2d_rotate.gif "KootenaiParticles")

## Description

This package advects conservative flow tracers (a.k.a. passive particles) with the fluid velocity and displaces them with stochastic diffusion due to fluid turbulence over discrete time steps. It tracks particles under a Lagrangian frame of reference as they move through a curvilinear 2- or 3-D hydrodynamic mesh. Users may customize particle subclasses to implement additional active particle motions, e.g. channel-bed adjacent sinusoidal vertical motion to simulate the preferred swimming patterns of white sturgeon larvae (McDonald and Nelson, 2021).

## Efficiently programmed and parallel enabled

As the total simulation duration, the size of the mesh, or the number of particles increases, so too do the computational resources used in the simulation (real-world time, memory, etc.). *fluvial-particle* uses the efficient array storage and operator methods of NumPy and VTK to update particle positions. Simulation results are written to hierarchical data format (HDF5) files using the h5py package, which allows writing and compression of terabytes of data. The mpi4py package enables massively-parallel execution mode to simulate millions or billions of particles (or more!). A strong-scaling test simulation of 2<sup>27</sup> particles on the Kootenai River over many thousands of CPUs shows that MPI-enabled fluvial-particle scales well, as shown in the figure below.

![Strong-scaling panel shows the decrease in simulation time and the simulation speed-up as a function of the number of CPUs (from 2^10 to 2^13 CPUs) used in the simulation. The scaling is very close to ideal over this range.](./data/strongscalingpanel.png "Parallel strong scaling")

For simulations over a large spatial or temporal domain, a large number of particles is needed to adequately resolve the particle concentrations. Consider the figure below. It shows a sequence of simulated particle density surfaces at the same time slice on the Kootenai River, Idaho, where each simulation uses a different number of particles. The left image shows the 2D output, and the right image shows the 3D surface and cross-sectional slice from the region outlined in the 2D image.
As the number of simulated particles increases, the 3D distribution taken from near the center of mass of the transported particles becomes continuous. The 2D distribution shows greater resolution as the number of particles increases, especially towards the tails of the distribution. For research in which capturing the distribution tails is important (e.g. river dye-dispersion studies), a sufficient number of particles must be used. Parallel execution with MPI enables greater particle concentration resolution without significantly increasing the total real-world execution time.

![An animated image shows a 2D and 3D snapshot of the Kootenai River results over multiple simulations that use different numbers of particles.](./data/kootenai_decimate_particles_fpc.gif "Simulation resolution as function of number of particles")


## Features

- TODO

## Requirements

To set up your development environment do the following.

- fork the reposistory

To remind yourself you're working on a fork.

```shell
git clone git@code.usgs.gov:<user_id>/fluvparticle.git fluvial-particle-fork
```

Setup Development Environment

```shell
conda env create -f environment.yml
conda develop -n fluvial-particle src
conda activate fluvial-particle
pip install -r requirements.dev
```

It is important to get [preccommit](https://pre-commit.com/) enabled on
the project, to ensure that certain standards are always met on a git
commit. With several of these, it might fail if files are changed, but
it will change them, and trying the commit a second time will actually
work.

### Git hook configuration

```shell
pre-commit install --install-hooks
```

### Testing

[Nox](https://nox.thea.codes/) is used for testing everything, with
several sessions built-in. To run the full suite of tests, simply use:

```shell
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

```shell
nox -s <session>
```

Replace `<session>` with the name of the session give above, i.e.:

```shell
nox -s mypy
```

You can also simply run [pytest](https://docs.pytest.org/en/latest/)
tests, by using the command:

```shell
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

```shell
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

```shell
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

You can install *fluvial-particle* via pip from [PyPI](https://pypi.org/):

```shell
pip install fluvial-particle
```
## Usage

Directions on invoking *fluvial-particle* from the command line can be found in the [docs](https://fluvial-particle.readthedocs.io/en/latest/usage.html).

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide](https://fluvial-particle.readthedocs.io/en/latest/contributing.html).

## License

Distributed under the terms of the [CCO 1.0 license](https://creativecommons.org/publicdomain/zero/1.0/legalcode), Fluvial Particle is free and open source software.

## Issues

If you encounter any problems,
please [file an issue](https://code.usgs.gov/wma/nhgf/fluvparticle/-/issues) along with a detailed description.

## Credits

This project was generated from [hillc-usgs's](https://github.com/hillc-usgs) [Pygeoapi Plugin Cookiecutter](https://code.usgs.gov/wma/nhgf/pygeoapi-plugin-cookiecutter) template.
