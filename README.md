# Welcome to _fluvial-particle_

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

![An animated image shows a fluvial-particle simulation output in the Kootenai River, Idaho, USA](https://code.usgs.gov/wma/nhgf/fluvparticle/-/raw/main/docs/data/kootenai_2to27_particles_fpc2d_rotate.gif "KootenaiParticles")

## Description

This package advects conservative flow tracers (a.k.a. passive particles) with the fluid velocity and displaces them with stochastic diffusion due to fluid turbulence over discrete time steps. It tracks particles under a Lagrangian frame of reference as they move through a curvilinear 2- or 3-D hydrodynamic mesh. Users may customize particle subclasses to implement additional active particle motions, e.g. channel-bed adjacent sinusoidal vertical motion to simulate the preferred swimming patterns of white sturgeon larvae (McDonald and Nelson, 2021).

## Efficiently programmed and parallel enabled

As the total simulation duration, the size of the mesh, or the number of particles increases, so too do the computational resources used in the simulation (real-world time, memory, etc.). _fluvial-particle_ uses the efficient array storage and operator methods of NumPy and VTK to update particle positions. Simulation results are written to hierarchical data format (HDF5) files using the h5py package, which allows writing and compression of terabytes of data.

Prohibitively large or long simulation problems can be made tractable with the highly scalable _fluvial-particle_ package. The mpi4py package enables massively-parallel execution mode to simulate millions or billions of particles (or more!). A strong-scaling test simulation of 2<sup>27</sup> particles on the Kootenai River over many thousands of CPUs shows that MPI-enabled fluvial-particle scales well, as shown in the figure below.

![Strong-scaling panel shows the decrease in simulation time and the simulation speed-up as a function of the number of CPUs (from 2^10 to 2^13 CPUs) used in the simulation. The scaling is very close to ideal over this range.](https://code.usgs.gov/wma/nhgf/fluvparticle/-/raw/main/docs/data/strongscalingpanel.png "Parallel strong scaling")

## Features

- TODO

## Installation

This package uses [poetry](https://python-poetry.org/) for installation and dependency management. It is recommended to use a conda environment.

First, create the conda environment with Python 3.9 and the latest version of poetry:

```shell
conda create -n fluvial-particle python==3.9 poetry -c conda-forge
conda activate fluvial-particle
```

Next, use poetry to install the package dependencies and _fluvial-particle_ itself:

```shell
poetry install
```

The success of the installation can be tested with pytest:

```shell
pytest tests
```

## Usage

Directions on invoking _fluvial-particle_ from the command line can be found in the [docs](https://fluvial-particle.readthedocs.io/en/latest/usage.html).

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

This package is based on the model described by [McDonald \& Nelson (2021)](https://doi.org/10.1080/24705357.2019.1709102), _A Lagrangian particle-tracking approach to modelling larval drift in rivers_, Journal of Ecohydraulics, 6(1) 17-35.
