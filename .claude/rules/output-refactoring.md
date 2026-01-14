# Working with Output Formats

## Available Output Formats

| Format | Files | Use Case |
|--------|-------|----------|
| HDF5 + XDMF | `particles.h5`, `particles.xmf` | Default, supports MPI, checkpointing |
| VTP + PVD | `vtp/*.vtp`, `particles.pvd` | Native ParaView, simpler visualization |

## Enabling VTP Output

Add to options file:
```python
output_vtp = True
```

Creates:
- `<output_dir>/vtp/particles_XXXX.vtp` - per-timestep files
- `<output_dir>/particles.pvd` - time series collection

## Key Output Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `VTPWriter` | `io/vtp_writer.py` | Write particle VTP files |
| `PVDWriter` | `io/pvd_writer.py` | Write PVD collection files |
| `Particles.create_hdf5()` | `Particles.py` | Create HDF5 output |
| `Particles.write_hdf5()` | `Particles.py` | Write timestep to HDF5 |
| `Particles.create_hdf5_xdmf()` | `Particles.py` | Generate XDMF metadata |

## Adding a New Output Format

1. Create writer class in `src/fluvial_particle/io/`
2. Export from `io/__init__.py`
3. Add initialization in `simulation.py` (after line ~140)
4. Add write calls in simulation loop
5. Add finalization before simulation end
6. Add setting option (e.g., `output_newformat = True`)
7. Document in `docs/optionsfile.rst` and `docs/output.rst`

## Output File Locations

```
<output_directory>/
├── particles.h5      # HDF5 particle data
├── particles.xmf     # XDMF metadata for ParaView
├── cells.h5          # Cell-centered post-processed data
├── cells_*.xmf       # Cell XDMF files (1D, 2D, 3D)
├── particles.pvd     # VTP collection (if output_vtp=True)
└── vtp/              # VTP files (if output_vtp=True)
    ├── particles_0000.vtp
    ├── particles_0001.vtp
    └── ...
```

## MPI Considerations

- HDF5 output uses `driver="mpio"` for parallel writes
- VTP output currently serial only (master rank)
- Postprocessing (`RiverGrid.postprocess()`) is serial only
