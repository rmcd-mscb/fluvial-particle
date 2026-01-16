"""General helper functions."""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
from os import getpid
from typing import Any

import h5py
import numpy as np
import vtk
from vtk.util import numpy_support


# Use tomllib (Python 3.11+) or tomli (backport)
if sys.version_info >= (3, 11):
    pass


# Template for user settings file
SETTINGS_TEMPLATE = '''\
"""User options file for fluvial-particle simulation.

This file defines all simulation parameters. Edit the values below to configure
your particle tracking simulation. See the documentation for detailed descriptions:
https://fluvial-particle.readthedocs.io/en/latest/optionsfile.html
"""

from fluvial_particle.Particles import Particles

# =============================================================================
# REQUIRED: Field name mappings
# =============================================================================
# Map standard field names to the names used in your mesh files.
# These depend on the hydrodynamic model that generated your mesh.

field_map_2d = {
    "bed_elevation": "Elevation",           # Bed/bottom elevation
    "shear_stress": "ShearStress (magnitude)",  # Shear stress magnitude
    "velocity": "Velocity",                 # Velocity vector
    "water_surface_elevation": "WaterSurfaceElevation",  # Water surface elevation
    # "wet_dry": "IBC",                     # Optional: wet/dry indicator (1=wet, 0=dry)
                                            # If omitted, computed from depth using min_depth
}

field_map_3d = {
    "velocity": "Velocity",                 # 3D velocity vector
}

# =============================================================================
# REQUIRED: Input mesh files
# =============================================================================
# Paths to your mesh files. Supported formats: .vts (recommended), .vtk, .npz

file_name_2d = "./path/to/your/mesh_2d.vts"
file_name_3d = "./path/to/your/mesh_3d.vts"

# =============================================================================
# REQUIRED: Simulation timing
# =============================================================================

SimTime = 60.0       # Maximum simulation time [seconds]
dt = 0.25            # Time step [seconds]
PrintAtTick = 10.0   # Output interval [seconds]

# =============================================================================
# REQUIRED: Simulation mode and particles
# =============================================================================

Track3D = 1          # 1 = 3D velocity field, 0 = 2D velocity field
NumPart = 100        # Number of particles per processor

# Starting location: tuple (x, y, z) or path to checkpoint file (.h5) or CSV (.csv)
StartLoc = (0.0, 0.0, 0.0)

# =============================================================================
# REQUIRED: Particle type
# =============================================================================
# Use Particles for basic transport, or subclasses for specialized behavior:
# - LarvalTopParticles, LarvalBotParticles: Sinusoidal swimming behavior
# - FallingParticles: Settling/falling particles

ParticleType = Particles  # noqa: F401

# =============================================================================
# OPTIONAL: Particle parameters
# =============================================================================

# startfrac = 0.5            # Initial vertical position as fraction of depth (0=bed, 1=surface)
# beta = (0.067, 0.067, 0.067)  # 3D diffusion coefficients
# lev = 0.25                 # Lateral eddy viscosity
# min_depth = 0.02           # Minimum depth threshold [meters]
# vertbound = 0.01           # Vertical boundary buffer (prevents particles at exact bed/surface)

# =============================================================================
# OPTIONAL: Output settings
# =============================================================================

# output_vtp = True          # Also write VTK PolyData (.vtp) files for ParaView

# =============================================================================
# OPTIONAL: Time-varying grids (for unsteady flow simulations)
# =============================================================================
# Uncomment and configure these to use time-dependent velocity fields.

# time_dependent = True
# file_pattern_2d = "./data/unsteady/Result_2D_{}.vts"  # {} = file index
# file_pattern_3d = "./data/unsteady/Result_3D_{}.vts"
# grid_start_index = 0       # First file index
# grid_end_index = 10        # Last file index (inclusive)
# grid_dt = 60.0             # Time between grid files [seconds]
# grid_start_time = 0.0      # Simulation time of first grid file
# grid_interpolation = "linear"  # "linear", "nearest", or "hold"

# =============================================================================
# OPTIONAL: LarvalParticles parameters (only for LarvalTopParticles/LarvalBotParticles)
# =============================================================================

# amp = 0.2                  # Amplitude of sinusoidal swimming
# period = 60.0              # Period of swimming behavior [seconds]

# =============================================================================
# OPTIONAL: FallingParticles parameters (only for FallingParticles)
# =============================================================================

# c1 = 20.0                  # Viscous drag coefficient
# c2 = 1.1                   # Turbulent wake drag coefficient
# radius = 0.0005            # Particle radius [meters]
# rho = 2650.0               # Particle density [kg/m^3]
'''

# TOML template for user settings file (recommended for new projects)
TOML_TEMPLATE = """\
# Fluvial-particle simulation configuration
# See: https://fluvial-particle.readthedocs.io/en/latest/optionsfile.html

# =============================================================================
# Simulation timing
# =============================================================================

[simulation]
time = 60.0              # Maximum simulation time [seconds]
dt = 0.25                # Time step [seconds]
print_interval = 10.0    # Output interval [seconds]

# =============================================================================
# Particle configuration
# =============================================================================

[particles]
type = "Particles"       # Options: Particles, FallingParticles, LarvalParticles,
                         #          LarvalTopParticles, LarvalBotParticles
count = 100              # Number of particles per processor
start_location = [0.0, 0.0, 0.0]  # Starting (x, y, z) coordinates
# start_depth_fraction = 0.5       # Optional: initial vertical position (0=bed, 1=surface)

[particles.physics]
beta = [0.067, 0.067, 0.067]  # 3D diffusion coefficients
lev = 0.25                    # Lateral eddy viscosity
min_depth = 0.02              # Minimum depth threshold [meters]
vertical_bound = 0.01         # Vertical boundary buffer

# Uncomment for FallingParticles:
# [particles.falling]
# radius = 0.0005       # Particle radius [meters]
# density = 2650.0      # Particle density [kg/m³]
# c1 = 20.0             # Viscous drag coefficient
# c2 = 1.1              # Turbulent wake drag coefficient

# Uncomment for LarvalParticles:
# [particles.larval]
# amplitude = 0.2       # Amplitude of sinusoidal swimming (fraction of depth)
# period = 60.0         # Period of swimming behavior [seconds]

# =============================================================================
# Grid configuration
# =============================================================================

[grid]
track_3d = true                              # true = 3D velocity field, false = 2D
file_2d = "./path/to/your/mesh_2d.vts"       # Path to 2D mesh file
file_3d = "./path/to/your/mesh_3d.vts"       # Path to 3D mesh file

# Field name mappings - map standard names to your model's field names
[grid.field_map_2d]
bed_elevation = "Elevation"
shear_stress = "ShearStress (magnitude)"
velocity = "Velocity"
water_surface_elevation = "WaterSurfaceElevation"
# wet_dry = "IBC"  # Optional: wet/dry indicator (auto-computed if omitted)

[grid.field_map_3d]
velocity = "Velocity"

# Optional: friction parameters for u* calculation
# [grid.friction]
# manning_n = 0.03      # Manning's n coefficient
# chezy_c = 50.0        # Chézy C coefficient
# darcy_f = 0.02        # Darcy-Weisbach f coefficient

# Optional: time-varying grids for unsteady flow
# [grid.time_varying]
# enabled = true
# file_pattern_2d = "./data/unsteady/Result_2D_{}.vts"  # {} = file index
# file_pattern_3d = "./data/unsteady/Result_3D_{}.vts"
# start_index = 0       # First file index
# end_index = 10        # Last file index (inclusive)
# dt = 60.0             # Time between grid files [seconds]
# interpolation = "linear"  # Options: linear, nearest, hold

# =============================================================================
# Output settings
# =============================================================================

[output]
vtp = false              # Also write VTP files for ParaView visualization
"""

# Default configuration as a nested dictionary (for programmatic use)
DEFAULT_CONFIG: dict[str, Any] = {
    "simulation": {
        "time": 60.0,
        "dt": 0.25,
        "print_interval": 10.0,
    },
    "particles": {
        "type": "Particles",
        "count": 100,
        "start_location": [0.0, 0.0, 0.0],
        "physics": {
            "beta": [0.067, 0.067, 0.067],
            "lev": 0.25,
            "min_depth": 0.02,
            "vertical_bound": 0.01,
        },
    },
    "grid": {
        "track_3d": True,
        "file_2d": "./path/to/your/mesh_2d.vts",
        "file_3d": "./path/to/your/mesh_3d.vts",
        "field_map_2d": {
            "bed_elevation": "Elevation",
            "shear_stress": "ShearStress (magnitude)",
            "velocity": "Velocity",
            "water_surface_elevation": "WaterSurfaceElevation",
        },
        "field_map_3d": {
            "velocity": "Velocity",
        },
    },
    "output": {
        "vtp": False,
    },
}


def get_default_config() -> dict[str, Any]:
    """Get the default configuration as a nested dictionary.

    Returns a deep copy of the default configuration that can be modified
    programmatically. This is the recommended way to create configurations
    in notebooks.

    Returns:
        Nested dictionary with all configuration options.

    Example::

        from fluvial_particle import get_default_config, run_simulation

        # Get default config and customize
        config = get_default_config()
        config["particles"]["count"] = 200
        config["simulation"]["time"] = 120.0
        config["grid"]["file_2d"] = "./my_mesh_2d.vts"
        config["grid"]["file_3d"] = "./my_mesh_3d.vts"

        # Run simulation directly with config dict
        results = run_simulation(config, output_dir="./output")
    """
    return copy.deepcopy(DEFAULT_CONFIG)


def save_config(config: dict[str, Any], path: str | pathlib.Path) -> None:
    """Save a configuration dictionary to a TOML file.

    Args:
        config: Nested configuration dictionary (as returned by get_default_config()).
        path: Path where the TOML file will be written.

    Example::

        from fluvial_particle import get_default_config, save_config

        config = get_default_config()
        config["particles"]["count"] = 200
        save_config(config, "my_settings.toml")
    """
    # We need to write TOML manually since tomllib is read-only
    # Use a simple recursive formatter
    lines = _dict_to_toml(config)
    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


def _dict_to_toml(d: dict[str, Any], prefix: str = "") -> list[str]:
    """Convert a nested dict to TOML format lines."""
    lines: list[str] = []
    tables: list[tuple[str, dict]] = []

    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Defer nested tables to write after scalar values
            tables.append((full_key, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")

    # Write nested tables
    for table_key, table_value in tables:
        lines.append("")
        lines.append(f"[{table_key}]")
        lines.extend(_dict_to_toml(table_value, table_key))

    return lines


def _toml_value(value: Any) -> str:
    """Format a Python value as a TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        items = ", ".join(_toml_value(v) for v in value)
        return f"[{items}]"
    if isinstance(value, (float, int)):
        return str(value)
    return repr(value)


def get_settings_template(format: str = "toml") -> str:
    """Get the settings template as a string.

    Returns the default settings template that can be customized and
    written to a file.

    Args:
        format: Template format - "toml" (default, recommended) or "python".

    Returns:
        Template string with all configuration options.

    Example::

        from fluvial_particle import get_settings_template
        from pathlib import Path

        # Get TOML template (recommended)
        template = get_settings_template()
        Path("my_settings.toml").write_text(template)

        # Or get Python template (legacy)
        py_template = get_settings_template(format="python")
    """
    if format == "toml":
        return TOML_TEMPLATE
    if format == "python":
        return SETTINGS_TEMPLATE
    raise ValueError(f"Unknown format: {format}. Use 'toml' or 'python'.")


def generate_settings_template(output_path: str | None = None, format: str = "toml") -> None:
    """Generate a template settings file for the user.

    Args:
        output_path: Path where the template file will be written.
                    Defaults to 'user_options.toml' (or .py for python format).
        format: Template format - "toml" (default, recommended) or "python".
    """
    if output_path is None:
        output_path = "user_options.toml" if format == "toml" else "user_options.py"

    output_file = pathlib.Path(output_path)

    if output_file.exists():
        print(f"Error: {output_file} already exists. Aborting to avoid overwriting.")
        raise SystemExit(1)

    template = get_settings_template(format=format)
    output_file.write_text(template, encoding="utf-8")
    print(f"Created settings template: {output_file}")
    print("\nNext steps:")
    print("  1. Edit the file to configure your simulation parameters")
    print("  2. Update file paths to point to your mesh files")
    print("  3. Adjust field_map_2d/field_map_3d for your hydrodynamic model")
    print(f"  4. Run: fluvial-particle {output_file} ./output")


def checkcommandarguments():
    """Check the user's command line arguments.

    Returns:
        dict: Parsed command-line arguments as a dictionary.

    Raises:
        SystemExit: If --init flag is provided (after generating template).
        FileNotFoundError: If settings_file does not exist.
        NotADirectoryError: If output_directory does not exist.
    """
    parser = create_parser()
    argdict = vars(parser.parse_args())

    # Handle --init flag
    if argdict.get("init"):
        generate_settings_template(format=argdict.get("format", "toml"))
        raise SystemExit(0)

    # Validate required positional arguments for simulation
    if argdict["settings_file"] is None:
        parser.error("settings_file is required (unless using --init)")
    if argdict["output_directory"] is None:
        parser.error("output_directory is required (unless using --init)")

    inputfile = pathlib.Path(argdict["settings_file"])
    if not inputfile.exists():
        raise FileNotFoundError(f"Cannot find settings file {inputfile}")
    outdir = pathlib.Path(argdict["output_directory"])
    if not outdir.is_dir():
        raise NotADirectoryError(f"Output directory {outdir} does not exist")

    return argdict


def convert_grid_hdf5tovtk(h5fname, output_dir, output_prefix="cells", output_threed=True):
    """Convert an HDF5 RiverGrid mesh output file into a time series of VTKStructuredGrid files.

    This function reads a specified HDF5 file containing grid data and converts it into multiple VTK files, either in
    2D or 3D format, based on the user's preference. The output files are named using a specified prefix and are saved
    in the designated output directory.

    Args:
        h5fname (str): Path to the RiverGrid HDF5 output file.
        output_dir (str): Directory to write output VTK files.
        output_prefix (str, optional): Shared name of the output VTK files. A suffix like 00.vtk will be appended
            to each one. Defaults to cells.
        output_threed (bool, optional): If True, output files will be on 3D grids. If False, output will be 2D.
            Defaults to True.

    Raises:
        NotADirectoryError: If the output directory output_dir does not exist.
    """
    outdir = pathlib.Path(output_dir)
    if not outdir.is_dir():
        raise NotADirectoryError(f"Output directory {outdir} does not exist")

    with h5py.File(h5fname, "r") as h5f:
        grid = h5f["grid"]
        n_prints = grid["time"].size  # the number of output files
        n_digits = len(str(n_prints - 1))  # the number of digits needed in file suffix
        if output_threed:
            x = grid["X"][()].ravel()
            y = grid["Y"][()].ravel()
            z = grid["Z"][()].ravel()
            dims = tuple(np.flip(grid["X"].shape))  # VTK uses x_i,y_j,z_k ordering
            grpname = "cells3d"
        else:
            x = grid["X"][0, ...].ravel()  # take just the z=0 slice
            y = grid["Y"][0, ...].ravel()
            z = np.zeros(x.size)  # VTK takes 3D points, even on a 2D structured grid
            dims = (grid["X"].shape[2], grid["X"].shape[1], 1)
            grpname = "cells2d"

        ptdata = np.stack([x, y, z]).T  # all the (x,y,z) grid points
        vptdata = numpy_support.numpy_to_vtk(ptdata)

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(x.size)
        pts.SetData(vptdata)

        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(dims)
        grid.SetPoints(pts)

        for j in range(n_prints):
            dname = "fpc" + str(j)
            data = h5f[grpname][dname][()].ravel()
            vdata = numpy_support.numpy_to_vtk(data)
            vdata.SetName("Fractional Particle Count")
            grid.GetCellData().AddArray(vdata)

            vtkout = output_prefix + f"{j:0{n_digits}d}" + ".vtk"
            vtkout = "/".join([output_dir, vtkout])
            writer = vtk.vtkStructuredGridWriter()
            writer.SetFileName(vtkout)
            writer.SetInputData(grid)
            writer.Write()

            grid.GetCellData().RemoveArray("Fractional Particle Count")


def convert_particles_hdf5tocsv(h5fname, output_dir, output_prefix="particles"):
    """Convert an HDF5 Particles output file into a time series of csv files.

    Args:
        h5fname (str): path to the Particles HDF5 output file
        output_dir (str): directory to write output csv files
        output_prefix (str, optional): shared name of the output csv files, a suffix like "00.csv" will be appended
            to each one. Defaults to "particles".

    Raises:
        NotADirectoryError: if the output directory output_dir does not exist
    """
    outdir = pathlib.Path(output_dir)
    if not outdir.is_dir():
        raise NotADirectoryError(f"Output directory {outdir} does not exist")

    with h5py.File(h5fname, "r") as h5f:
        coords = h5f["coordinates"]
        props = h5f["properties"]

        x = coords["x"][()]
        y = coords["y"][()]
        z = coords["z"][()]
        time = coords["time"][()]
        bedelev = props["bedelev"][()]
        cellidx2d = props["cellidx2d"][()]
        cellidx3d = props["cellidx3d"][()]
        depth = props["depth"][()]
        htabvbed = props["htabvbed"][()]
        velvec = props["velvec"][()]
        wse = props["wse"][()]
        vx = velvec[..., 0]
        vy = velvec[..., 1]
        vz = velvec[..., 2]

        n_prints = time.size  # the number of output files
        n_digits = len(str(n_prints - 1))  # the number of digits needed in file suffix

        for j in range(n_prints):
            csv_out = output_prefix + f"{j:0{n_digits}d}" + ".csv"
            csv_out = "/".join([output_dir, csv_out])
            with pathlib.Path(csv_out).open("w", encoding="utf-8") as f:
                # write header first using "w" flag to overwrite existing file
                header = [
                    "time",
                    "x",
                    "y",
                    "z",
                    "bed_elevation",
                    "cell_index_2d",
                    "cell_index_3d",
                    "depth",
                    "height_above_bed",
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                    "water_surface_elevation",
                ]
                f.write(", ".join(header) + "\n")
            with pathlib.Path(csv_out).open("a", encoding="utf-8") as f:
                # now write data to same file in append mode
                idx = np.s_[j, :]
                t = time[idx] + np.zeros(x[idx].shape)
                data = np.stack([
                    t,
                    x[idx],
                    y[idx],
                    z[idx],
                    bedelev[idx],
                    cellidx2d[idx],
                    cellidx3d[idx],
                    depth[idx],
                    htabvbed[idx],
                    vx[idx],
                    vy[idx],
                    vz[idx],
                    wse[idx],
                ]).T
                np.savetxt(f, data, delimiter=",")


def create_parser():
    """Factory method to create an argument parser for command-line arguments.

    Returns:
        argparse.ArgumentParser: the container for command line argument specifications
    """
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="fluvial-particle",
        description=(
            "Lagrangian particle tracking for fluvial environments. "
            "Simulates particle transport in rivers using velocity fields "
            "from hydrodynamic models (Delft-FM, iRIC, HEC-RAS, etc.)."
        ),
        epilog=(
            "Example usage:\n"
            "  fluvial-particle settings.toml ./output    Run simulation\n"
            "  fluvial-particle --init                    Generate TOML template\n"
            "  fluvial-particle --init --format python    Generate Python template (legacy)\n"
            "  fluvial-particle --version                 Show version information\n"
            "\n"
            "Documentation: https://fluvial-particle.readthedocs.io/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Generate a template settings file in the current directory.",
    )

    parser.add_argument(
        "--format",
        dest="format",
        choices=["toml", "python"],
        default="toml",
        help="Format for --init template: 'toml' (default, recommended) or 'python' (legacy).",
    )

    parser.add_argument(
        "settings_file",
        nargs="?",
        help=(
            "Path to the settings file (.toml recommended, or .py for legacy). "
            "See documentation for required and optional parameters."
        ),
    )

    parser.add_argument(
        "output_directory",
        nargs="?",
        help=(
            "Directory where output files (particles.h5, particles.xmf, cells.h5, etc.) "
            "will be written. Must exist before running."
        ),
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Random seed for reproducible simulations. If not specified, "
            "a seed is generated from the current time and process ID. "
            "Only used in serial mode."
        ),
    )

    parser.add_argument(
        "--no-postprocess",
        "--no_postprocess",
        action="store_false",
        help="Skip RiverGrid post-processing (XDMF file generation, cell counters).",
    )  # argparse converts dest to "no_postprocess"; store_false means default=True

    return parser


def get_prng(timer, comm=None, seed=None):
    """Generate a random seed using time and the process id, then create and return the random number generator.

    Args:
        timer (time.time or MPI.Wtime): object that controls the timing; time.time for serial execution, MPI.Wtime
            for parallel
        comm (MPI.Intracomm): MPI communicator for parallel execution. Defaults to None
        seed (int): random seed

    Returns:
        np.random.RandomState: the random number generator
    """
    if seed is None:
        seed = np.int64(np.abs(((timer() * 181) * ((getpid() - 83) * 359)) % 104729))

    if comm is None:
        print(f"Using seed {seed}", flush=True)

    return np.random.RandomState(seed)


def load_checkpoint(fname, tidx, start, end, comm=None):
    """Load initial positions from a checkpoint HDF5 file.

    This function retrieves the starting positions of particles from a specified checkpoint file in HDF5
    format. It supports parallel execution using MPI, allowing for efficient data loading across multiple
    processors.

    Args:
        fname (str): Path to the checkpoint HDF5 file.
        tidx (int): Outer index of HDF5 datasets, indicating the specific time step to load.
        start (int): Starting index of this processor's assigned space.
        end (int): Ending index (non-inclusive) for the data slice.
        comm (mpi4py communicator, optional): MPI communicator for parallel runs. If None, the function runs in a
            single process.

    Returns:
        Tuple(ndarray, ndarray, ndarray, int): A tuple containing the (x, y, z) starting positions of the
            particles and the simulation start time.

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
    """
    if comm is None or comm.Get_rank() == 0:
        print("Loading initial particle positions from a checkpoint HDF5 file")
    inputfile = pathlib.Path(fname)
    if not inputfile.exists():
        raise FileNotFoundError(f"Cannot find load_checkpoint HDF5 file: {fname}")
    h5file = h5py.File(fname, "r") if comm is None else h5py.File(fname, "r", driver="mpio", comm=comm)

    grp = h5file["coordinates"]
    x = grp["x"][tidx, start:end]
    y = grp["y"][tidx, start:end]
    z = grp["z"][tidx, start:end]
    t = grp["time"][tidx].item(0)  # returns t as a Python basic float

    h5file.close()

    return x, y, z, t


def load_variable_source(
    fname: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load variable source data.

    Input file must be a comma separated values file with 5 columns:

        #. start_time (float): the time at which to activate the particles
        #. x(float): the starting x-coordinate of the particles
        #. y(float): the starting y-coordinate of the particles
        #. z(float): the starting z-coordinate of the particles
        #. numpart (int): the number of particles to activate

    Each row will add additional particles to the simulation.
    For example, if a given row in the CSV file is "10.0, 6.14, 9.09, 10.3, 100", then 100 particles will be initiated
    from the point (6.14, 9.09, 10.3) starting at a simulation time of 10.0 seconds.

    Args:
        fname (str): path to CSV file containing the variable source data

    Raises:
        FileNotFoundError: the path to the input CSV given in fname is not valid
        ValueError: the input did not contain 5 columns per row

    Returns:
        Tuple(ndarray, ndarray, ndarray, ndarray): each output ndarray is 1D and has length equal to the summed numpart
        column
    """
    inputfile = pathlib.Path(fname)
    if not inputfile.exists():
        raise FileNotFoundError(f"Cannot find variable source file: {fname}")
    data = np.genfromtxt(inputfile, delimiter=",")
    if data.shape[1] != 5:
        raise ValueError("Expected 5 columns in variable source file(start_time, x, y, z, numpart)")
    numparts = np.int64(np.sum(data[:, 4]))
    pstime = np.zeros(numparts, dtype=np.int64)
    x = np.zeros(numparts, dtype=np.float64)
    y = np.zeros(numparts, dtype=np.float64)
    z = np.zeros(numparts, dtype=np.float64)
    count = 0
    for i in np.arange(data.shape[0]):
        npart = np.int64(data[i, 4])
        for _j in np.arange(npart):
            pstime[count] = data[i, 0]
            x[count] = data[i, 1]
            y[count] = data[i, 2]
            z[count] = data[i, 3]
            count += 1

    return pstime, x, y, z
