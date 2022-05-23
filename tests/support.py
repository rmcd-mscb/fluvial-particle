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
    return f["coordinates"].get("x").shape[0]


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
    pt_dim = f["coordinates"].get("x").shape[1]
    x = f["coordinates"].get("x")
    y = f["coordinates"].get("y")
    z = f["coordinates"].get("z")
    if not twod:
        return np.array(
            [[[x[i, j], y[i, j], z[i, j]] for i in [time] for j in np.arange(pt_dim)]][
                0
            ]
        )
    else:
        # returns z as 0.5 so it sits above 2d mesh
        return np.array(
            [[[x[i, j], y[i, j], 0.5] for i in [time] for j in np.arange(pt_dim)]][0]
        )
