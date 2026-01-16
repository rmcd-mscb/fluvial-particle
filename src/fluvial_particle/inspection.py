"""Grid inspection utilities for exploring input data.

This module provides convenience functions for inspecting and summarizing
grid data from user options files, particularly useful in Jupyter notebooks
before running simulations.

Example usage::

    from fluvial_particle import inspect_grid

    # Basic usage - loads grid and prints summary
    info = inspect_grid("user_options.py")

    # For time-dependent grids, specify timestep
    info = inspect_grid("user_options.py", timestep=5)

    # Access returned data
    print(info["grid_2d"]["dimensions"])
    print(info["hydraulics"]["depth"]["mean"])
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import vtk
from vtk.util import numpy_support

from .RiverGrid import (
    CMU,
    DEFAULT_WATER_DENSITY,
    GRAVITY,
    RiverGrid,
)
from .Settings import Settings


def _read_raw_vtk_grid(filename: str):
    """Read a VTK grid file without any processing.

    Args:
        filename: Path to VTK/VTS file.

    Returns:
        VTK structured grid object with all original arrays.
    """
    suffix = pathlib.Path(filename).suffix.lower()
    if suffix == ".vts":
        reader = vtk.vtkXMLStructuredGridReader()
    elif suffix == ".vtk":
        reader = vtk.vtkStructuredGridReader()
    else:
        # For unsupported formats, return None
        return None

    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def inspect_grid(
    settings_file: str | pathlib.Path,
    *,
    timestep: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Inspect grid data from a user options file.

    Loads the grid(s) specified in the options file and returns a summary
    of dimensions, extents, available fields, and reach-averaged hydraulic
    statistics.

    Args:
        settings_file: Path to the user settings file (Python script).
        timestep: For time-dependent grids, which timestep index to inspect (0-indexed).
                 Defaults to 0 (first timestep). Negative values supported (-1 = last).
        quiet: If True, suppress printed summary. Default False.

    Returns:
        Dictionary containing grid information with keys:
            - grid_2d: 2D grid info (file, dimensions, extents, scalars, vectors)
            - grid_3d: 3D grid info (only if Track3D=1)
            - hydraulics: Reach-averaged statistics (depth, velocity, shear_stress, ustar)
            - time_dependent: Boolean or timestep info dict
            - ustar_method: The method used to compute shear velocity

    Raises:
        FileNotFoundError: If settings file or grid files don't exist.

    Example::

        from fluvial_particle import inspect_grid

        info = inspect_grid("settings.py")
        print(f"Grid dimensions: {info['grid_2d']['dimensions']}")
        print(f"Mean depth: {info['hydraulics']['depth']['mean']:.2f} m")
    """
    settings_path = pathlib.Path(settings_file)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")

    # Load settings
    settings = Settings.read(str(settings_path))

    # Check if time-dependent
    time_dependent = settings.get("time_dependent", False)

    # Determine which grid files to load
    if time_dependent:
        grid_info = _inspect_time_dependent(settings, timestep)
    else:
        grid_info = _inspect_static(settings)

    # Print summary if not quiet
    if not quiet:
        _print_summary(grid_info)

    return grid_info


def _extract_ustar_options(settings: dict) -> dict[str, Any]:
    """Extract u* configuration options from settings.

    Args:
        settings: Settings dictionary.

    Returns:
        Dictionary with keys: manning_n, chezy_c, darcy_f, water_density,
        ustar_method, min_depth (values may be None if not configured).
    """
    return {
        "manning_n": settings.get("manning_n"),
        "chezy_c": settings.get("chezy_c"),
        "darcy_f": settings.get("darcy_f"),
        "water_density": settings.get("water_density"),
        "ustar_method": settings.get("ustar_method"),
        "min_depth": settings.get("min_depth"),
    }


def _inspect_static(settings: dict) -> dict[str, Any]:
    """Inspect static (single timestep) grid files."""
    track3d = settings["Track3D"]
    file_2d = settings["file_name_2d"]
    file_3d = settings.get("file_name_3d")
    field_map_2d = settings["field_map_2d"]
    field_map_3d = settings["field_map_3d"]

    # Extract u* configuration options
    ustar_opts = _extract_ustar_options(settings)

    # Read the raw 2D grid BEFORE RiverGrid processes it (to get velocity for stats)
    raw_grid_2d = _read_raw_vtk_grid(file_2d)

    # Load the grid
    river = RiverGrid(
        track3d=track3d,
        filename2d=file_2d,
        filename3d=file_3d if track3d else None,
        field_map_2d=field_map_2d,
        field_map_3d=field_map_3d,
        **ustar_opts,
    )

    # Extract grid info from raw grid (before processing removes arrays)
    result = {
        "grid_2d": _extract_grid_info(raw_grid_2d, file_2d, is_3d=False),
        "hydraulics": _compute_hydraulics(river, settings, raw_grid_2d=raw_grid_2d),
        "time_dependent": False,
        "ustar_method": river._ustar_method,
    }

    if track3d and river.vtksgrid3d is not None:
        result["grid_3d"] = _extract_grid_info(river.vtksgrid3d, file_3d, is_3d=True)

    return result


def _inspect_time_dependent(settings: dict, timestep: int | None) -> dict[str, Any]:
    """Inspect time-dependent grid files."""
    from .grids import TimeVaryingGrid

    track3d = settings["Track3D"]
    file_pattern_2d = settings["file_pattern_2d"]
    file_pattern_3d = settings["file_pattern_3d"]
    field_map_2d = settings["field_map_2d"]
    field_map_3d = settings["field_map_3d"]
    grid_start_index = settings["grid_start_index"]
    grid_end_index = settings["grid_end_index"]
    grid_dt = settings["grid_dt"]
    grid_start_time = settings.get("grid_start_time", 0.0)

    # Extract u* configuration options
    ustar_opts = _extract_ustar_options(settings)

    # Calculate the actual timestep index
    n_grids = grid_end_index - grid_start_index + 1
    original_timestep = timestep
    if timestep is None:
        timestep = 0
    elif timestep < 0:
        timestep = n_grids + timestep

    if timestep < 0 or timestep >= n_grids:
        raise ValueError(
            f"timestep {original_timestep} out of range for {n_grids} available timesteps "
            f"(valid range: 0 to {n_grids - 1}, or -{n_grids} to -1)"
        )

    # Load the time-varying grid
    river = TimeVaryingGrid(
        track3d=track3d,
        file_pattern_2d=file_pattern_2d,
        file_pattern_3d=file_pattern_3d,
        field_map_2d=field_map_2d,
        field_map_3d=field_map_3d,
        grid_start_index=grid_start_index,
        grid_end_index=grid_end_index,
        grid_dt=grid_dt,
        grid_start_time=grid_start_time,
        **ustar_opts,
    )

    # Advance to the requested timestep if needed
    target_time = river.grid_times[timestep]
    if timestep > 0:
        river.advance_to_time(target_time)

    # Get the actual file paths for this timestep
    file_idx = grid_start_index + timestep
    file_2d = file_pattern_2d.format(file_idx)
    file_3d = file_pattern_3d.format(file_idx) if track3d else None

    # Read raw 2D grid to get all fields including velocity
    raw_grid_2d = _read_raw_vtk_grid(file_2d)

    result = {
        "grid_2d": _extract_grid_info(raw_grid_2d, file_2d, is_3d=False),
        "hydraulics": _compute_hydraulics(river._current_grid, settings, raw_grid_2d=raw_grid_2d),
        "time_dependent": {
            "enabled": True,
            "timestep": timestep,
            "time": target_time,
            "total_timesteps": n_grids,
            "grid_dt": grid_dt,
            "time_range": (river.grid_times[0], river.grid_times[-1]),
        },
        "ustar_method": river._current_grid._ustar_method,
    }

    if track3d and river._current_grid.vtksgrid3d is not None:
        result["grid_3d"] = _extract_grid_info(river._current_grid.vtksgrid3d, file_3d, is_3d=True)

    return result


def _extract_grid_info(vtk_grid, filename: str, is_3d: bool) -> dict[str, Any]:
    """Extract dimensions, extents, and field info from a VTK grid."""
    # Get dimensions - VTK fills in the list argument
    dims = [0, 0, 0]
    vtk_grid.GetDimensions(dims)
    if is_3d:
        dimensions = {"i": dims[0], "j": dims[1], "k": dims[2]}
    else:
        dimensions = {"i": dims[0], "j": dims[1]}

    # Get extents from points
    bounds = vtk_grid.GetBounds()
    extents = {
        "x": (bounds[0], bounds[1]),
        "y": (bounds[2], bounds[3]),
        "z": (bounds[4], bounds[5]),
    }

    # Get field names
    point_data = vtk_grid.GetPointData()
    scalars = []
    vectors = []

    for i in range(point_data.GetNumberOfArrays()):
        arr = point_data.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        n_components = arr.GetNumberOfComponents()

        if n_components == 1:
            scalars.append(name)
        elif n_components == 3:
            vectors.append(name)

    return {
        "file": str(filename),
        "dimensions": dimensions,
        "extents": extents,
        "scalars": scalars,
        "vectors": vectors,
        "num_points": vtk_grid.GetNumberOfPoints(),
        "num_cells": vtk_grid.GetNumberOfCells(),
    }


def _compute_hydraulics(
    river: RiverGrid,
    settings: dict,
    raw_grid_2d=None,
) -> dict[str, dict[str, float]]:
    """Compute reach-averaged hydraulic statistics from the 2D grid.

    Computes statistics only for wet cells (where wet_dry > 0).

    Args:
        river: RiverGrid object (post-processed).
        settings: Settings dictionary.
        raw_grid_2d: Optional raw VTK grid for reading velocity before processing.
    """
    ptdata = river.vtksgrid2d.GetPointData()

    # Get wet mask
    wet_dry_arr = ptdata.GetArray("wet_dry")
    if wet_dry_arr is not None:
        wet_dry = numpy_support.vtk_to_numpy(wet_dry_arr)
        wet_mask = wet_dry > 0
    else:
        # If no wet_dry, assume all points are wet
        wet_mask = np.ones(river.vtksgrid2d.GetNumberOfPoints(), dtype=bool)

    if not np.any(wet_mask):
        # No wet points - return zeros
        return {
            "depth": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "velocity_mag": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "shear_stress": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "ustar": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
        }

    result = {}

    # Depth
    bed_elev_arr = ptdata.GetArray("bed_elevation")
    wse_arr = ptdata.GetArray("water_surface_elevation")
    if bed_elev_arr is not None and wse_arr is not None:
        bed_elev = numpy_support.vtk_to_numpy(bed_elev_arr)
        wse = numpy_support.vtk_to_numpy(wse_arr)
        depth = np.maximum(wse - bed_elev, 0.0)
        depth_wet = depth[wet_mask]
        result["depth"] = _compute_stats(depth_wet)
    else:
        result["depth"] = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    # Velocity magnitude - check processed grid first, then raw grid
    vel_arr = ptdata.GetArray("velocity")
    if vel_arr is None and raw_grid_2d is not None:
        # Use raw grid to get velocity (before RiverGrid processing removes it)
        raw_ptdata = raw_grid_2d.GetPointData()
        # Try common velocity array names
        for name in ("Velocity", "velocity", "vel", "U"):
            vel_arr = raw_ptdata.GetArray(name)
            if vel_arr is not None and vel_arr.GetNumberOfComponents() == 3:
                break
        else:
            vel_arr = None

    if vel_arr is not None and vel_arr.GetNumberOfComponents() >= 2:
        vel = numpy_support.vtk_to_numpy(vel_arr)
        vel_mag = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
        vel_mag_wet = vel_mag[wet_mask]
        result["velocity_mag"] = _compute_stats(vel_mag_wet)
    else:
        result["velocity_mag"] = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    # Shear stress (if available)
    shear_arr = ptdata.GetArray("shear_stress")
    if shear_arr is not None:
        shear = numpy_support.vtk_to_numpy(shear_arr)
        shear_wet = shear[wet_mask]
        result["shear_stress"] = _compute_stats(shear_wet)
    else:
        result["shear_stress"] = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    # Compute u* based on the method
    result["ustar"] = _compute_ustar_stats(river, settings, wet_mask, raw_grid_2d=raw_grid_2d)

    return result


def _compute_ustar_stats(
    river: RiverGrid,
    settings: dict,
    wet_mask: np.ndarray,
    raw_grid_2d=None,
) -> dict[str, float]:
    """Compute u* statistics based on the configured method."""
    ptdata = river.vtksgrid2d.GetPointData()
    method = river._ustar_method

    # Get necessary arrays
    bed_elev_arr = ptdata.GetArray("bed_elevation")
    wse_arr = ptdata.GetArray("water_surface_elevation")
    vel_arr = ptdata.GetArray("velocity")

    if bed_elev_arr is None or wse_arr is None:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    bed_elev = numpy_support.vtk_to_numpy(bed_elev_arr)
    wse = numpy_support.vtk_to_numpy(wse_arr)
    depth = np.maximum(wse - bed_elev, 1e-6)

    # Try processed grid first, then raw grid for velocity
    if vel_arr is None and raw_grid_2d is not None:
        raw_ptdata = raw_grid_2d.GetPointData()
        for name in ("Velocity", "velocity", "vel", "U"):
            vel_arr = raw_ptdata.GetArray(name)
            if vel_arr is not None and vel_arr.GetNumberOfComponents() == 3:
                break
        else:
            vel_arr = None

    if vel_arr is not None and vel_arr.GetNumberOfComponents() >= 2:
        vel = numpy_support.vtk_to_numpy(vel_arr)
        vel_mag = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
    else:
        vel_mag = np.zeros_like(depth)

    # Compute u* based on method
    if method == "ustar":
        ustar_arr = ptdata.GetArray("ustar")
        if ustar_arr is not None:
            ustar = numpy_support.vtk_to_numpy(ustar_arr)
        else:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    elif method == "shear_stress":
        shear_arr = ptdata.GetArray("shear_stress")
        if shear_arr is not None:
            shear = numpy_support.vtk_to_numpy(shear_arr)
            rho = settings.get("water_density", DEFAULT_WATER_DENSITY)
            ustar = np.sqrt(np.maximum(shear, 0.0) / rho)
        else:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    elif method == "manning":
        manning_arr = ptdata.GetArray("manning_n")
        if manning_arr is not None:
            n = numpy_support.vtk_to_numpy(manning_arr)
        else:
            n = settings.get("manning_n", 0.03)
        ustar = vel_mag * n * np.sqrt(GRAVITY) / np.power(depth, 1.0 / 6.0)

    elif method == "chezy":
        chezy_arr = ptdata.GetArray("chezy_c")
        if chezy_arr is not None:
            c = numpy_support.vtk_to_numpy(chezy_arr)
        else:
            c = settings.get("chezy_c", 50.0)
        c = np.maximum(c, 1e-6)
        ustar = vel_mag * np.sqrt(GRAVITY) / c

    elif method == "darcy":
        darcy_arr = ptdata.GetArray("darcy_f")
        if darcy_arr is not None:
            f = numpy_support.vtk_to_numpy(darcy_arr)
        else:
            f = settings.get("darcy_f", 0.02)
        ustar = vel_mag * np.sqrt(np.maximum(f, 0.0) / 8.0)

    elif method == "energy_slope":
        slope_arr = ptdata.GetArray("energy_slope")
        if slope_arr is not None:
            slope = numpy_support.vtk_to_numpy(slope_arr)
            ustar = np.sqrt(GRAVITY * depth * np.maximum(slope, 0.0))
        else:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    elif method == "tke":
        tke_arr = ptdata.GetArray("tke")
        if tke_arr is not None:
            tke = numpy_support.vtk_to_numpy(tke_arr)
            ustar = np.power(CMU, 0.25) * np.sqrt(np.maximum(tke, 0.0))
        else:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    else:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    ustar = np.maximum(ustar, 0.0)
    ustar_wet = ustar[wet_mask]
    return _compute_stats(ustar_wet)


def _compute_stats(arr: np.ndarray) -> dict[str, float]:
    """Compute basic statistics for an array."""
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _print_summary(info: dict[str, Any]) -> None:
    """Print a formatted summary of grid information."""
    lines = ["", "Grid Summary", "============"]

    # 2D Grid info
    g2d = info["grid_2d"]
    lines.append(f"\n2D Grid: {g2d['file']}")
    dims = g2d["dimensions"]
    lines.append(f"  Dimensions (i, j): {dims['i']} x {dims['j']}")
    lines.append(f"  Points: {g2d['num_points']:,}  Cells: {g2d['num_cells']:,}")
    lines.append("  Extents:")
    ext = g2d["extents"]
    lines.append(f"    x: [{ext['x'][0]:.2f}, {ext['x'][1]:.2f}] m")
    lines.append(f"    y: [{ext['y'][0]:.2f}, {ext['y'][1]:.2f}] m")
    lines.append(f"    z: [{ext['z'][0]:.2f}, {ext['z'][1]:.2f}] m")

    # 3D Grid info (if present)
    if "grid_3d" in info:
        g3d = info["grid_3d"]
        lines.append(f"\n3D Grid: {g3d['file']}")
        dims = g3d["dimensions"]
        lines.append(f"  Dimensions (i, j, k): {dims['i']} x {dims['j']} x {dims['k']}")
        lines.append(f"  Points: {g3d['num_points']:,}  Cells: {g3d['num_cells']:,}")
        lines.append("  Extents:")
        ext = g3d["extents"]
        lines.append(f"    x: [{ext['x'][0]:.2f}, {ext['x'][1]:.2f}] m")
        lines.append(f"    y: [{ext['y'][0]:.2f}, {ext['y'][1]:.2f}] m")
        lines.append(f"    z: [{ext['z'][0]:.2f}, {ext['z'][1]:.2f}] m")

    # Time-dependent info
    td = info["time_dependent"]
    if td and isinstance(td, dict):
        lines.append("\nTime-Dependent: Yes")
        lines.append(f"  Timestep: {td['timestep'] + 1} of {td['total_timesteps']}")
        lines.append(f"  Time: {td['time']:.2f} s")
        lines.append(f"  Grid dt: {td['grid_dt']:.2f} s")
        lines.append(f"  Time range: [{td['time_range'][0]:.2f}, {td['time_range'][1]:.2f}] s")

    # Available fields
    lines.append("\nAvailable Fields (2D):")
    lines.append(f"  Scalars: {', '.join(g2d['scalars'])}")
    lines.append(f"  Vectors: {', '.join(g2d['vectors'])}")

    if "grid_3d" in info:
        g3d = info["grid_3d"]
        lines.append("\nAvailable Fields (3D):")
        lines.append(f"  Scalars: {', '.join(g3d['scalars']) if g3d['scalars'] else '(none)'}")
        lines.append(f"  Vectors: {', '.join(g3d['vectors'])}")

    # u* method
    lines.append(f"\nu* Method: {info['ustar_method']}")

    # Hydraulic statistics
    hyd = info["hydraulics"]
    lines.append("\nReach-Averaged Hydraulics (wet cells only):")

    def fmt_stat(name: str, stats: dict, unit: str) -> str:
        if np.isnan(stats["mean"]):
            return f"  {name:15} (not available)"
        return (
            f"  {name:15} {stats['mean']:8.3f} +/- {stats['std']:.3f} {unit}  [{stats['min']:.3f}, {stats['max']:.3f}]"
        )

    lines.append(fmt_stat("Depth:", hyd["depth"], "m"))
    lines.append(fmt_stat("Velocity mag:", hyd["velocity_mag"], "m/s"))
    lines.append(fmt_stat("Shear stress:", hyd["shear_stress"], "Pa"))
    lines.append(fmt_stat("u*:", hyd["ustar"], "m/s"))

    print("\n".join(lines))
