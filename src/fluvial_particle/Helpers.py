"""General helper functions."""
import pathlib
from typing import Tuple

import numpy as np


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
        Tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray): each output ndarray is 1D and has length numpart
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
