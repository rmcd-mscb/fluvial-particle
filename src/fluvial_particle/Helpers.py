"""General helper functions."""

import argparse
import pathlib
from os import getpid
from typing import Tuple

import h5py
import numpy as np
import vtk
from vtk.util import numpy_support


def checkcommandarguments():
    """Check the user's command line arguments."""
    parser = create_parser()

    argdict = vars(parser.parse_args())

    inputfile = pathlib.Path(argdict["settings_file"])
    if not inputfile.exists():
        raise FileNotFoundError(f"Cannot find settings file {inputfile}")
    outdir = pathlib.Path(argdict["output_directory"])
    if not outdir.is_dir():
        raise NotADirectoryError(f"Output directory {outdir} does not exist")

    return argdict


def convert_grid_hdf5tovtk(
    h5fname, output_dir, output_prefix="cells", output_threed=True
):
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
        htbabvbed = props["htabvbed"][()]
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
            with open(csv_out, "w") as f:
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
            with open(csv_out, "a") as f:
                # now write data to same file in append mode
                idx = np.s_[j, :]
                t = time[idx] + np.zeros(x[idx].shape)
                data = np.stack(
                    [
                        t,
                        x[idx],
                        y[idx],
                        z[idx],
                        bedelev[idx],
                        cellidx2d[idx],
                        cellidx3d[idx],
                        depth[idx],
                        htbabvbed[idx],
                        vx[idx],
                        vy[idx],
                        vz[idx],
                        wse[idx],
                    ]
                ).T
                np.savetxt(f, data, delimiter=",")


def create_parser():
    """Factory method to create an argument parser for command-line arguments.

    Returns:
        argparse.ArgumentParser: the container for command line argument specifications
    """
    parser = argparse.ArgumentParser(
        description="fluvial_particle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("settings_file", help="User settings file")
    parser.add_argument("output_directory", help="Output directory for results")
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Specify a single integer to fix the seed of the random number generator. Only used in serial mode.",
    )
    parser.add_argument(
        "--no-postprocess",
        "--no_postprocess",
        action="store_false",
        help="Include this flag to prevent RiverGrid post-processing.",
    )  # note: argparse will convert to key="no_postprocess"

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
    if comm is None:
        h5file = h5py.File(fname, "r")
    else:
        h5file = h5py.File(fname, "r", driver="mpio", comm=comm)

    grp = h5file["coordinates"]
    x = grp["x"][tidx, start:end]
    y = grp["y"][tidx, start:end]
    z = grp["z"][tidx, start:end]
    t = grp["time"][tidx].item(0)  # returns t as a Python basic float

    h5file.close()

    return x, y, z, t


def load_variable_source(
    fname: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load variable source data.

    Input file must be a comma separated values file with 5 columns:

        #. start_time (float): the time at which to activate the particles
        #. x(float): the starting x-coordinate of the particles
        #. y(float): the starting y-coordinate of the particles
        #. z(float): the starting z-coordinate of the particles
        #. numpart (int): the number of particles to activate

    Each row will add additional particles to the simulation.
    For example, if a given row in the CSV file is “10.0, 6.14, 9.09, 10.3, 100”, then 100 particles will be iniated
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
        raise ValueError(
            (
                "Expected 5 columns in variable source file"
                "(start_time, x, y, z, numpart)"
            )
        )
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
