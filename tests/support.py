"""Helper functions for tests."""

from typing import Optional

import h5py
import numpy as np


def get_h5file(filename: str) -> h5py._hl.files.File:
    """Get and open h5 file.

    Args:
        filename (str): path to HDF5 file

    Returns:
        h5py._hl.files.File: the opened HDF5 file object
    """
    return h5py.File(filename)


def get_num_timesteps(f: h5py._hl.files.File) -> int:
    """Get number of timesteps.

    Args:
        f (h5py._hl.files.File): the open HDF5 file object

    Returns:
        int: the number of timesteps in the file datasets
    """
    return f["coordinates"]["x"].shape[0]


def get_points(
    f: h5py._hl.files.File, time: int, twod: Optional[bool] = False
) -> np.ndarray:
    """Get point coordinates at time-step time.

    Args:
        f (h5py._hl.files.File): the open HDF5 file object
        time (int): the time index to slice into the datasets
        twod (Optional[bool], optional): use False to return 2D point coordinates or True to return 3D
            point coordinates. If 2D, all z values will be set to 0.5. Defaults to False.

    Returns:
        np.ndarray: point coordinates at the selected time slice
    """
    # f = h5py.File(filename)
    x = f["coordinates"]["x"][time, :]
    y = f["coordinates"]["y"][time, :]
    z = f["coordinates"]["z"][time, :]
    if not twod:
        return np.stack([x, y, z]).T
    else:
        return np.stack([x, y, 0.5 * np.ones(x.size)]).T
