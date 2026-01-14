# Working with Input File Formats

## Supported Grid Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| VTK XML Structured Grid | `.vts` | **Preferred** - Binary with compression |
| VTK Legacy | `.vtk` | ASCII or binary, widely compatible |
| NumPy Archive | `.npz` | Python-specific, fixed field names |

## Field Name Mapping

Different hydrodynamic models use different array names. Map model names to standard internal names using `field_map_2d` and `field_map_3d` in the options file.

### Required 2D Fields
- `bed_elevation` - bed/bottom elevation
- `shear_stress` - shear stress magnitude
- `velocity` - velocity vector (3 components)
- `water_surface_elevation` - water surface elevation

### Optional 2D Fields
- `wet_dry` - wet/dry indicator (1=wet, 0=dry). If omitted, auto-computed from depth using `min_depth`.

### Required 3D Fields
- `velocity` - velocity vector (3 components)

### Example Field Map
```python
field_map_2d = {
    "bed_elevation": "Elevation",
    "shear_stress": "ShearStress (magnitude)",
    "velocity": "Velocity",
    "water_surface_elevation": "WaterSurfaceElevation",
    # wet_dry optional - omit to auto-compute
}
field_map_3d = {"velocity": "Velocity"}
```

## Adding a New Input Format

1. Add reader method in `RiverGrid.py` (see `read_2d_data()`, `read_3d_data()`)
2. Handle file extension detection in the suffix check
3. Apply field mapping with `_apply_field_mapping()`
4. Add tests with sample data in `tests/data/`

## Time-Varying Grids

For unsteady simulations, provide a file sequence instead of single files:

```python
time_dependent = True
file_pattern_2d = "./data/flow_2d_{}.vts"  # {} = file index
file_pattern_3d = "./data/flow_3d_{}.vts"
grid_start_index = 2
grid_end_index = 6
grid_dt = 1.0  # seconds between files
grid_interpolation = "linear"  # or "nearest", "hold"
```

The `TimeVaryingGrid` class handles:
- Sliding window loading (2 grids in memory)
- Automatic advancement during simulation
- Temporal interpolation between timesteps

## NPZ Format Details

NPZ files use fixed internal names (no field mapping):
- Coordinates: `x`, `y`, `z` (optional)
- Scalars: `elev`, `wse`, `shear`
- Velocity: `vx`, `vy`, `vz` (optional)
- Wet/dry: `ibc` (optional)
