"""Helper functions for tests."""
from typing import Optional

import h5py
import numpy as np


def get_h5file(filename: str) -> h5py._hl.files.File:
    """Get and open h5 file.

    Args:
        filename (str): _description_

    Returns:
        h5py._hl.files.File: _description_
    """
    return h5py.File(filename)


def get_num_timesteps(f: h5py._hl.files.File) -> int:
    """Get number of timesteps.

    Args:
        f (h5py._hl.files.File): _description_

    Returns:
        int: _description_
    """
    return f["coordinates"]["x"].shape[0]


def get_points(
    f: h5py._hl.files.File, time: int, twod: Optional[bool] = False
) -> np.ndarray:
    """Get point coordinates at time-step time.

    Args:
        f (h5py._hl.files.File): _description_
        time (int): _description_
        twod (Optional[bool], optional): _description_. Defaults to False.

    Returns:
        np.ndarray: _description_
    """
    # f = h5py.File(filename)
    x = f["coordinates"]["x"][time, :]
    y = f["coordinates"]["y"][time, :]
    z = f["coordinates"]["z"][time, :]
    if not twod:
        return np.stack([x, y, z]).T
    else:
        return np.stack([x, y, 0.5 * np.ones(x.size)]).T
