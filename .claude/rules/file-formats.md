# Supported Input File Formats

## Grid Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| VTK XML Structured Grid | `.vts` | **Recommended** - Binary with compression |
| VTK Legacy | `.vtk` | ASCII or binary, widely compatible |
| NumPy Archive | `.npz` | Python-specific compressed format |

## Field Name Mapping

Different hydrodynamic models use different array names. Use `field_map_2d` and `field_map_3d` to map model names to standard internal names.

### Standard 2D Fields (required)
- `bed_elevation` - bed/bottom elevation
- `shear_stress` - shear stress magnitude
- `velocity` - velocity vector
- `water_surface_elevation` - water surface elevation

### Optional 2D Fields
- `wet_dry` - wet/dry indicator (1=wet, 0=dry). If omitted, computed from depth using `min_depth` threshold.

### Standard 3D Fields
- `velocity` - velocity vector

### Example (Delft-FM)

```python
field_map_2d = {
    "bed_elevation": "Elevation",
    "wet_dry": "IBC",  # optional
    "shear_stress": "ShearStress (magnitude)",
    "velocity": "Velocity",
    "water_surface_elevation": "WaterSurfaceElevation",
}
field_map_3d = {
    "velocity": "Velocity",
}
```

## Time-Varying Grids (PR #22)

For time-dependent simulations, use file sequences instead of single grid files:

```python
time_dependent = True
file_pattern_2d = "./data/flow_2d_{}.vts"  # {} replaced with index
file_pattern_3d = "./data/flow_3d_{}.vts"
grid_start_index = 2      # First file index
grid_end_index = 6        # Last file index
grid_dt = 1.0             # Seconds between grid files
grid_interpolation = "linear"  # linear | nearest | hold
```

The `TimeVaryingGrid` class in `src/fluvial_particle/grids/` manages:
- Sliding window loading (2 grids in memory)
- Automatic grid advancement during simulation
- Temporal interpolation between grid timesteps

## VTS Time Metadata

VTS files can include time metadata in FieldData:
- `TimeValue` (Float64): Time in seconds for ParaView
- `TimeStep` (Int32): Integer timestep index

## NPZ Format

For `.npz` files, field mappings are not used. NPZ uses fixed internal names:
- `x`, `y`, `z` (optional) - coordinates
- `elev`, `wse`, `shear` - scalar fields
- `vx`, `vy`, `vz` (optional) - velocity components
- `ibc` (optional) - wet/dry indicator
