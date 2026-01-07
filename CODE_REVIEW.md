# Fluvial-Particle Code Review & Development Roadmap

**Date:** 2026-01-07
**Reviewer:** Claude Code
**Version Reviewed:** 0.0.3

## Project Overview

**Fluvial-Particle** is a well-engineered scientific computing package for Lagrangian particle tracking in rivers. It efficiently models active and passive particle transport in flowing rivers with support for:

- Advection with fluid velocity
- Stochastic diffusion modeling
- 2D and 3D velocity fields
- Custom particle subclasses (sediment, larvae)
- MPI parallelization (2²⁷ particles tested)
- HDF5/XDMF output for visualization

---

## Code Quality Assessment

### Strengths ✓

- **Well-structured architecture** with clear separation of concerns
- **Extensible design** with particle subclassing system
- **Strong testing infrastructure** with integration and unit tests
- **Comprehensive documentation** with Sphinx
- **Modern Python tooling** (Poetry, Nox, pre-commit hooks)
- **MPI parallelization** with proven scalability
- **Type hints** with strict mypy checking

### Areas for Improvement

- Test coverage could be increased (currently 60% minimum)
- Some modules are large (Particles.py is 1,509 lines)
- ReadTheDocs build currently failing (noted in commit 797d908)
- Limited support for time-varying velocity fields
- No GPU acceleration for large-scale simulations

---

## Suggestions for Further Development

### 1. Testing & Quality Assurance

**Current Coverage: 60% minimum**

- [ ] **Increase test coverage** to 80%+ for critical modules
  - Focus on `Particles.py`, `RiverGrid.py`, and `Helpers.py`
  - Add tests for edge cases: empty particle sets, zero timesteps, boundary conditions

- [ ] **Add property-based testing** with `hypothesis` for numerical algorithms
  - Test particle advection with randomized inputs
  - Verify conservation laws and physical constraints

- [ ] **Add performance regression tests** to detect slowdowns
  - Benchmark critical operations (interpolation, diffusion)
  - Alert on >10% performance degradation

- [ ] **Add stress tests** for memory usage with large particle counts
  - Test with 10⁶, 10⁷, 10⁸ particles
  - Verify memory scaling is linear

- [ ] **Mock VTK dependencies** in unit tests to improve test speed
  - Use fixtures for VTK grid objects
  - Reduce test runtime by 50%+

### 2. Performance Enhancements

- [ ] **GPU acceleration** via CuPy or Numba
  - Target: 10-100x speedup for particle advection/diffusion
  - Particularly beneficial for `move()` and `interp_fields()` methods
  - Consider CUDA kernels for VTK interpolation alternative

- [ ] **JIT compilation** with Numba for hot loops
  - Annotate performance-critical functions in `Particles.py`
  - Focus on `calc_diffusion_coefs()` and array operations

- [ ] **Optimize interpolation** by caching VTK probe results
  - Spatial hashing for nearby particles
  - Reuse probe filter objects across timesteps

- [ ] **Implement adaptive timestep** based on local velocity gradients
  - Courant–Friedrichs–Lewy (CFL) condition
  - Automatic timestep adjustment for accuracy/performance

- [ ] **Profile and optimize** `interp_fields()` method
  - Likely bottleneck in simulations
  - Consider vectorized VTK operations

### 3. New Features

#### High Priority

- [ ] **Particle-particle interactions**
  - Collision detection for dense particle clouds
  - K-d tree or octree spatial indexing
  - Schooling behavior for larvae (Reynolds flocking rules)

- [ ] **Time-varying velocity fields**
  - Support loading multiple timesteps of hydrodynamic data
  - Temporal interpolation between velocity snapshots
  - Memory-efficient streaming of large datasets
  - Support for NetCDF/ADCIRC time series

- [ ] **Particle sources & sinks**
  - Continuous injection at specified locations/times
  - Particle removal at domain boundaries or zones
  - Source strength variations (tidal, diurnal patterns)

- [ ] **Improved boundary conditions**
  - Reflective boundaries (currently particles just stop)
  - Periodic boundaries for idealized studies
  - Wall-interaction models (no-slip, partial slip)
  - Deposition/resuspension for sediment particles

#### Medium Priority

- [ ] **3D visualization tools**
  - Built-in ParaView state file (.pvsm) generation
  - Automated rendering scripts for animations
  - Trajectory streamlines and density plots

- [ ] **Statistical analysis module**
  - Particle residence time calculations
  - Dispersion coefficient estimation (Fickian vs. anomalous)
  - Connectivity matrices between regions
  - Exposure time analysis for ecological applications

- [ ] **Support for unstructured meshes**
  - Currently requires structured VTK grids
  - Would enable more complex river geometries
  - Support for ANUGA, SELFE/SCHISM, ADCIRC formats

- [ ] **Temperature-dependent behavior**
  - Swimming speed variations with water temperature
  - Settling velocity temperature dependence
  - Load temperature fields from hydrodynamic model

#### Low Priority

- [ ] **Web-based configuration GUI** for non-programmers
  - Flask/FastAPI backend
  - React frontend with form validation
  - Job submission and monitoring

- [ ] **Real-time visualization** during simulation
  - WebSocket streaming to browser
  - Progress monitoring and early termination

- [ ] **Export to other formats**
  - NetCDF (CF-conventions compliant)
  - Zarr for cloud-native workflows
  - GeoJSON for GIS integration

### 4. Code Architecture Improvements

- [ ] **Refactor `Particles.py` (1,509 lines)**
  - Split into multiple modules:
    - `advection.py` - Velocity interpolation and advection
    - `diffusion.py` - Stochastic diffusion calculations
    - `io.py` - HDF5 reading/writing
    - `boundary.py` - Boundary condition handling

- [ ] **Add abstract base class** for particle types
  ```python
  from abc import ABC, abstractmethod

  class BaseParticle(ABC):
      @abstractmethod
      def move(self, dt):
          pass

      @abstractmethod
      def initialize(self, num_particles):
          pass
  ```

- [ ] **Dependency injection** for RNG to improve testability
  - Pass `numpy.random.Generator` objects instead of global state
  - Enable reproducible testing with fixed seeds

- [ ] **Configuration validation** with Pydantic
  - Type-safe settings with automatic validation
  - Better error messages for invalid configurations
  - JSON schema generation for documentation

- [ ] **Logging framework** using `logging` module
  - Replace print statements with proper logging
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR)
  - Log file output with rotation

- [ ] **Add dataclasses** for structured parameter groups
  ```python
  from dataclasses import dataclass

  @dataclass
  class DiffusionParams:
      diffusivity_x: float
      diffusivity_y: float
      eddy_viscosity: float
  ```

### 5. Documentation Enhancements

- [ ] **Tutorial notebooks** showing:
  - How to create custom particle subclasses
  - Coupling with hydrodynamic models (Delft3D, TELEMAC, HEC-RAS)
  - Post-processing and visualization workflows with ParaView
  - Sensitivity analysis and parameter calibration

- [ ] **Performance tuning guide**
  - Optimal particle count per MPI rank
  - Memory usage estimation formulas
  - Scaling efficiency on different HPC architectures
  - Profiling and optimization tips

- [ ] **Contributing guide** (CONTRIBUTING.md)
  - Code style requirements (Black, Flake8)
  - How to add new particle types
  - Pull request process
  - Testing requirements

- [ ] **Example gallery** with real-world case studies
  - White sturgeon larvae in Columbia River
  - Sediment transport in meandering channels
  - Pollutant dispersion scenarios

- [ ] **API stability guarantees**
  - Semantic versioning policy
  - Deprecation timeline (2 major versions)
  - Breaking change documentation

### 6. Infrastructure & DevOps

- [ ] **Refactor to use uv and hatchling instead of Poetry**
  - Modern, faster dependency resolver with `uv`
  - Hatchling provides simpler, standards-based build backend
  - Benefits:
    - 10-100x faster dependency resolution than Poetry
    - PEP 621 compliant (metadata in `pyproject.toml`)
    - Better integration with modern Python packaging ecosystem
    - Simpler build configuration with fewer abstractions
  - Migration steps:
    - Convert `pyproject.toml` from Poetry to Hatchling format
    - Replace `poetry.lock` with `uv.lock`
    - Update CI/CD pipelines to use `uv pip install`
    - Update documentation (installation instructions)

- [ ] **Fix ReadTheDocs build** (noted in commit 797d908)
  - Check Sphinx configuration
  - Verify environment.yml dependencies
  - Test local build with `sphinx-build`

- [ ] **Add benchmarking suite** with `pytest-benchmark`
  - Track performance over time
  - Compare different algorithms
  - CI/CD integration for regression detection

- [ ] **Automated releases** via GitHub Actions
  - Publish to PyPI on git tag
  - Generate release notes from commits
  - Build and upload documentation

- [ ] **Conda-forge recipe** for easier installation
  - Simplify dependencies (VTK, mpi4py)
  - Support multiple platforms (Linux, macOS, Windows)

- [ ] **Docker container** for reproducible environments
  - Base image with all dependencies
  - Example: `docker run -v $(pwd):/data fluvial-particle config.py`

- [ ] **Add CITATION.cff** for academic citations
  - Machine-readable citation format
  - Link to McDonald & Nelson (2021) paper

### 7. Scientific Enhancements

- [ ] **Validate against analytical solutions**
  - Taylor-Green vortex for 2D advection
  - Gaussian plume for diffusion
  - Quantify numerical errors and convergence rates

- [ ] **Comparison with other tools**
  - OpenDrift, Ichthyop, Parcels, PTM
  - Benchmark accuracy and performance
  - Document differences in algorithms

- [ ] **Uncertainty quantification**
  - Ensemble runs with parameter variations
  - Monte Carlo analysis tools
  - Sensitivity analysis (Sobol indices)

- [ ] **Particle age/tagging system**
  - Track particle origin and time since release
  - Color-code by source location
  - Age-based particle removal

### 8. User Experience

- [ ] **Progress bars** using `tqdm` for long simulations
  ```python
  from tqdm import tqdm
  for t in tqdm(range(num_timesteps), desc="Simulating"):
      particles.move(dt)
  ```

- [ ] **Better error messages** with recovery suggestions
  - Validate inputs before simulation start
  - Suggest fixes for common errors
  - Link to documentation for complex issues

- [ ] **Configuration templates** for common scenarios
  - `examples/passive_tracer.py`
  - `examples/sediment_transport.py`
  - `examples/larval_drift.py`

- [ ] **Validation mode** to check inputs before running
  - `fluvial-particle --validate config.py`
  - Verify mesh files exist and are readable
  - Check parameter ranges and compatibility

- [ ] **Resume from failure** with automatic checkpointing
  - Save state every N timesteps
  - `--resume` flag to continue from last checkpoint
  - Graceful handling of SIGTERM/SIGINT

---

## Priority Roadmap

### Immediate (Next Release)

1. **Fix ReadTheDocs build** ⚠️
   - Critical for documentation access
   - Check environment.yml and conf.py

2. **Increase test coverage to 75%**
   - Add tests for `RiverGrid` methods
   - Test boundary condition handling

3. **Add time-varying velocity fields support**
   - High user demand
   - Essential for realistic simulations

4. **Improve error messages and validation**
   - Low effort, high impact
   - Better user experience

### Short-term (3-6 months)

1. **GPU acceleration prototype**
   - Evaluate CuPy vs. Numba
   - Target 10x speedup for large simulations

2. **Tutorial notebooks**
   - Lower barrier to entry
   - Demonstrate key features

3. **Particle source/sink functionality**
   - Continuous release scenarios
   - Boundary condition improvements

4. **Statistical analysis module**
   - Residence time, connectivity
   - Post-processing automation

### Long-term (6-12 months)

1. **Unstructured mesh support**
   - Major architectural change
   - Enables complex geometries

2. **Particle-particle interactions**
   - Advanced physics
   - Schooling behaviors

3. **Web-based GUI**
   - Accessibility for non-coders
   - Configuration and visualization

4. **Comprehensive benchmark suite**
   - Validation and verification
   - Performance tracking

---

## Quick Wins (Low Effort, High Impact)

These can be implemented quickly and provide immediate value:

1. **Add progress bars** - 30 minutes
   - `pip install tqdm`, add to move() loop
   - Huge UX improvement

2. **Logging framework** - 2 hours
   - Replace print() with logging.info()
   - Better debugging and log management

3. **Configuration templates** - 1 hour
   - Copy existing test configs to examples/
   - Add inline comments explaining options

4. **CITATION.cff** - 30 minutes
   - Copy template, fill in metadata
   - Proper attribution for users

5. **Docker container** - 2 hours
   - Write Dockerfile with dependencies
   - Push to Docker Hub for easy access

---

## Metrics for Success

Track these metrics to measure progress:

- **Test coverage**: 60% → 80%
- **Documentation completeness**: API reference, tutorials, examples
- **Performance**: Particles/second/core on standard benchmark
- **User adoption**: Downloads, citations, GitHub stars
- **Code quality**: Mypy strict mode, Flake8 compliance
- **Issue resolution time**: Average time to close issues
- **MPI scaling efficiency**: Strong scaling to 1000+ cores

---

## References

- McDonald & Nelson (2021). "Fluvial-Particle: A Python package for Lagrangian particle tracking in rivers." Journal of Ecohydraulics.
- Repository: https://code.usgs.gov/vtcfwru/fluvial-particle
- Documentation: https://fluvial-particle.readthedocs.io/

---

## Notes

This review is based on codebase analysis as of 2026-01-07. The project demonstrates strong software engineering practices and scientific rigor. Focus areas for improvement include:

- Enhanced testing and validation
- Performance optimization (GPU, JIT)
- Time-varying velocity fields
- User experience improvements

The codebase is well-positioned for continued development and broader adoption in the river ecology and sediment transport communities.
