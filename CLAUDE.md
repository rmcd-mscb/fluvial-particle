# Claude Code Instructions for fluvial-particle

Particle tracking in river flows using VTK structured grids.

## Quick Reference

```bash
# Always use uv run for commands
uv run pytest              # run tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy src/           # type check
uv run sphinx-build docs docs/_build/html  # build docs
```

## Project Structure

```
src/fluvial_particle/
├── RiverGrid.py      # Grid loading and field mapping
├── Particles.py      # Particle classes and movement
├── simulation.py     # Main simulation loop
├── Settings.py       # Options file parsing
└── Helpers.py        # Utilities and I/O
```

## Key Concepts

- **Field mapping**: Maps model-specific array names to standard internal names
- **Supported formats**: `.vts` (recommended), `.vtk`, `.npz`
- **Standard fields (2D)**: `bed_elevation`, `wet_dry` (optional), `shear_stress`, `velocity`, `water_surface_elevation`
- **wet_dry auto-compute**: If omitted from field_map_2d, computed from depth using `min_depth` threshold

## Modular Rules

Detailed instructions are in `.claude/rules/`:
- `environment.md` - uv setup and commands
- `testing.md` - pytest and code quality
- `documentation.md` - Sphinx and ReadTheDocs
- `versioning.md` - bump-my-version workflow
- `file-formats.md` - VTK/VTS/NPZ and field mapping
- `notebooks.md` - Jupyter notebook editing
