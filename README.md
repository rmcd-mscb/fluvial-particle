# Welcome to _fluvial-particle_

[![PyPI](https://img.shields.io/pypi/v/fluvial-particle.svg)](https://pypi.org/project/fluvial-particle/)
[![Status](https://img.shields.io/pypi/status/fluvial-particle.svg)](https://pypi.org/project/fluvial-particle/)
[![Python Version](https://img.shields.io/pypi/pyversions/fluvial-particle)](https://pypi.org/project/fluvial-particle/)
[![License](https://img.shields.io/pypi/l/fluvial-particle)](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
[![Read the Docs](https://img.shields.io/readthedocs/fluvial-particle/latest.svg?label=Read%20the%20Docs)](https://fluvial-particle.readthedocs.io/)

[![Tests](https://github.com/rmcd-mscb/fluvial-particle/actions/workflows/tests.yml/badge.svg)](https://github.com/rmcd-mscb/fluvial-particle/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/rmcd-mscb/fluvial-particle/branch/main/graph/badge.svg)](https://codecov.io/gh/rmcd-mscb/fluvial-particle)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python package to efficiently model active- and passive-particle transport in flowing rivers.

![An animated image shows a fluvial-particle simulation output in the Kootenai River, Idaho, USA](https://raw.githubusercontent.com/rmcd-mscb/fluvial-particle/main/docs/data/kootenai_2to27_particles_fpc2d_rotate.gif "KootenaiParticles")

## Description

This package advects conservative flow tracers (a.k.a. passive particles) with the fluid velocity and displaces them with stochastic diffusion due to fluid turbulence over discrete time steps. It tracks particles under a Lagrangian frame of reference as they move through a curvilinear 2- or 3-D hydrodynamic mesh. Users may customize particle subclasses to implement additional active particle motions, e.g. channel-bed adjacent sinusoidal vertical motion to simulate the preferred swimming patterns of white sturgeon larvae (McDonald and Nelson, 2021). Since version 0.1.0 the package also tracks particles along 1D river networks routed by [pywatershed](https://github.com/EC-USGS/pywatershed), described in the section below.

## Efficiently programmed and parallel enabled

As the total simulation duration, the size of the mesh, or the number of particles increases, so too do the computational resources used in the simulation (real-world time, memory, etc.). _fluvial-particle_ uses the efficient array storage and operator methods of NumPy and VTK to update particle positions. Simulation results are written to hierarchical data format (HDF5) files using the h5py package, which allows writing and compression of terabytes of data.

Prohibitively large or long simulation problems can be made tractable with the highly scalable _fluvial-particle_ package. The mpi4py package enables massively-parallel execution mode to simulate millions or billions of particles (or more!). A strong-scaling test simulation of 2<sup>27</sup> particles on the Kootenai River over many thousands of CPUs shows that MPI-enabled fluvial-particle scales well, as shown in the figure below.

![Strong-scaling panel shows the decrease in simulation time and the simulation speed-up as a function of the number of CPUs (from 2^10 to 2^13 CPUs) used in the simulation. The scaling is very close to ideal over this range.](https://raw.githubusercontent.com/rmcd-mscb/fluvial-particle/main/docs/data/strongscalingpanel.png "Parallel strong scaling")

## Features

- Lagrangian particle tracking on 2D and 3D hydrodynamic meshes (VTK `.vts` and `.vtk`, NumPy `.npz`), with active-particle subclasses for behaviors such as larval swimming and settling
- 1D river-network particle tracking driven by [pywatershed](https://github.com/EC-USGS/pywatershed) network hydraulics exports (new in 0.1.0, see below)
- Parallel execution with MPI through mpi4py
- TOML settings files, a notebook API (`run_simulation` / `SimulationResults` and `run_network_simulation` / `NetworkResults`), and HDF5/XDMF, VTP/PVD, and NetCDF output

## Installation

This package uses [uv](https://github.com/astral-sh/uv) for fast dependency installation and [hatchling](https://hatch.pypa.io/) for building. It is recommended to use a conda environment.

First, create the fluvial-particle conda environment using the environment.yml file:

```shell
conda env create -f environment.yml
conda activate fluvial-particle
```

Next, install uv if you don't have it already:

```shell
pip install uv
```

Then use uv to install the package and its dependencies:

```shell
uv pip install -e .
```

The success of the installation can be tested with pytest:

```shell
pytest tests
```

## Usage

Directions on invoking _fluvial-particle_ from the command line can be found in the [docs](https://fluvial-particle.readthedocs.io/en/latest/usage.html).

## 1D river-network particle tracking

Version 0.1.0 adds a second solver, `fluvial_particle.network`, for passive transport along a river network rather than across a 2D/3D mesh. It reads the network hydraulics NetCDF export written by [pywatershed](https://github.com/EC-USGS/pywatershed) (`pywatershed.utils.export_network_hydraulics`): reach topology, daily flow, velocity, depth, width and shear velocity, and optional map polylines. Particles advect exactly along reaches, cross junctions carrying the unused part of the time step, disperse with a Fischer coefficient built from the exported hydraulics, and exit at outlets with their arrival times recorded. Sources are mass loadings (slugs, constant or tabulated loading, or concentration curves), and the results give arrival-time distributions, map positions, and concentration on sub-reach bins.

A minimal settings file:

```toml
[network]
hydraulics_file = "drb_network_hydraulics.nc"
start_time = "1979-03-01"
end_time = "1979-04-01"
dt = 900.0
output_interval = 3600.0
particle_mass = 1.0

[[network.sources]]
reach_id = 4205
form = "slug"
time = 0.0
mass = 1000.0
```

Run it from the command line or from Python:

```shell
fluvial_particle_network settings.toml -o output
mpiexec -n 4 fluvial_particle_network_mpi settings.toml -o output
```

```python
from fluvial_particle import run_network_simulation

res = run_network_simulation("settings.toml", "output", seed=42)
print(res.summary())
arrivals = res.arrival_histogram(outlet=4205, bin_seconds=3600)
concentration = res.concentration(time=-1, bin_length=500.0)
```

See the [network documentation](https://fluvial-particle.readthedocs.io/en/latest/network.html) and the demo notebooks: `notebooks/network-drb-demo.ipynb` (Delaware River Basin: headwater slugs, a continuous loading, breakthrough at Trenton, a map animation) and `notebooks/network-chain-dispersion-demo.ipynb` (a uniform chain compared with the analytical advection-dispersion solution).

## Utilities

- Environments with [conda](https://www.anaconda.com) plus [uv](https://github.com/astral-sh/uv) for fast installs; packaging with [hatchling](https://hatch.pypa.io/)
- Test automation with [Nox](https://nox.thea.codes/); testing with [pytest](https://docs.pytest.org/) and [Coverage.py](https://coverage.readthedocs.io/), reported to [Codecov](https://codecov.io/)
- Linting, formatting, and [pre-commit](https://pre-commit.com/) hooks with [Ruff](https://docs.astral.sh/ruff/); static type checking with [mypy](http://mypy-lang.org/)
- Security checks with [Bandit](https://github.com/PyCQA/bandit) and [pip-audit](https://github.com/pypa/pip-audit)
- Continuous integration, releases, and labels with [GitHub Actions](https://github.com/features/actions), [Release Drafter](https://github.com/release-drafter/release-drafter), and [GitHub Labeler](https://github.com/marketplace/actions/github-labeler)
- Documentation with [Sphinx](http://www.sphinx-doc.org/) ([autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html), [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html), [MyST](https://myst-parser.readthedocs.io/)) on [Read the Docs](https://readthedocs.org/)
- Version management with [bump-my-version](https://github.com/callowayproject/bump-my-version)

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide](https://fluvial-particle.readthedocs.io/en/latest/contributing.html).

## License

Distributed under the terms of the [CCO 1.0 license](https://creativecommons.org/publicdomain/zero/1.0/legalcode), Fluvial Particle is free and open source software.

## Issues

If you encounter any problems,
please [file an issue](https://github.com/rmcd-mscb/fluvial-particle/issues) along with a detailed description.

## Credits

This project was generated from [hillc-usgs's](https://github.com/hillc-usgs) [Pygeoapi Plugin Cookiecutter](https://code.usgs.gov/wma/nhgf/pygeoapi-plugin-cookiecutter) template.

This package is based on the model described by [McDonald \& Nelson (2021)](https://doi.org/10.1080/24705357.2019.1709102), _A Lagrangian particle-tracking approach to modelling larval drift in rivers_, Journal of Ecohydraulics, 6(1) 17-35.
