"""General helper functions."""
import pathlib
from typing import Tuple

import numpy as np


def load_variable_source(
    fname: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load variable source data.

    Args:
        fname (str): _description_

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    inputfile = pathlib.Path(fname)
    if not inputfile.exists():
        raise Exception(f"Cannot find variable source file: {fname}")
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
