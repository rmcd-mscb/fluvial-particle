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
        raise Exception(f"Cannot find settings file {inputfile}")
    outdir = pathlib.Path(argdict["output_directory"])
    if not outdir.is_dir():
        raise Exception(f"Output directory {outdir} does not exist")

    return argdict


def convert_grid_hdf5tovtk(h5fname, output_dir, output_prefix="cells", output_threed=True):
    """Convert a HDF5 RiverGrid mesh file into a time series of VTKStructuredGrid files.

    Args:
        h5fname (str): path to the RiverGrid HDF5 output file
        output_dir (str): directory to write output VTK files
        output_prefix (str, optional): shared name of the output VTK files, a suffix like "00.vtk" will be appended to each one.
            Defaults to "cells".
        output_threed (bool, optional): if True, output files will be on 3D grids. If False, output will be 2D. Defaults to True.

    Raises:
        Exception: if the output directory output_dir does not exist
    """
    outdir = pathlib.Path(output_dir)
    if not outdir.is_dir():
        raise Exception(f"Output directory {outdir} does not exist")

    with h5py.File(h5fname,"r") as h5f:
        grid = h5f["grid"]
        n_prints = grid["time"].size  # the number of output files = number of time steps
        n_digits = len(str(n_prints - 1))
        if output_threed:
            X = grid["X"][()].ravel()  # raveled because VTK takes grid points as a collection of (x,y,z) tuples
            Y = grid["Y"][()].ravel()
            Z = grid["Z"][()].ravel()
            dims = tuple(np.flip(grid["X"].shape))  # VTK uses x_i,y_j,z_k ordering
            grpname = "cells3d"
        else:
            X = grid["X"][0, ...].ravel()  # take just the z=0 slice
            Y = grid["Y"][0, ...].ravel()
            Z = np.zeros(X.size)  # VTK takes 3D points, even on a 2D structured grid
            dims = (grid["X"].shape[2], grid["X"].shape[1], 1)
            grpname = "cells2d"

        ptdata = np.stack([X,Y,Z]).T  # all the (x,y,z) grid points
        vptdata = numpy_support.numpy_to_vtk(ptdata)

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(X.size)
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
        timer (time.time or MPI.Wtime): object that controls the timing; time.time for serial execution, MPI.Wtime for parallel
        comm (MPI.Intracomm): MPI communicator for parallel execution. Defaults to None
        seed (int): random seed

    Returns:
        np.random.RandomState: the random number generator

    """
    if seed is None:
        seed = np.int64(np.abs(((timer() * 181) * ((getpid() - 83) * 359)) % 104729))

    if comm is None:
        print(f"Using seed {seed}", flush=True)

    prng = np.random.RandomState(seed)
    return prng


def load_checkpoint(fname, tidx, start, end, comm=None):
    """Load initial positions from a checkpoint HDF5 file.

    Args:
        fname (str): path to checkpoint HDF5 file
        tidx (int): outer index of HDF5 datasets
        start (int): starting index of this processor's assigned space
        end (int): ending index (non-inclusive)
        comm (mpi4py communicator, optional): for parallel runs.

    Returns:
        Tuple(ndarray, ndarray, ndarray, int): the (x,y,z) starting positions of the particles and the simulation start time
    """
    if comm is None or comm.Get_rank() == 0:
        print("Loading initial particle positions from a checkpoint HDF5 file")
    inputfile = pathlib.Path(fname)
    if not inputfile.exists():
        raise Exception(f"Cannot find load_checkpoint HDF5 file: {fname}")
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
        Exception: the input did not contain 5 columns per row

    Returns:
        Tuple(ndarray, ndarray, ndarray, ndarray): each output ndarray is 1D and has length equal to the summed numpart column
    """
    inputfile = pathlib.Path(fname)
    if not inputfile.exists():
        raise FileNotFoundError(f"Cannot find variable source file: {fname}")
    data = np.genfromtxt(inputfile, delimiter=",")
    if not data.shape[1] == 5:
        raise Exception(
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
