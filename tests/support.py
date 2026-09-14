"""Helper functions for tests."""

import pathlib

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


def get_points(f: h5py._hl.files.File, time: int, twod: bool | None = False) -> np.ndarray:
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
    return np.stack([x, y, 0.5 * np.ones(x.size)]).T


def write_straight_channel(
    dirpath,
    *,
    length: float = 600.0,
    width: float = 100.0,
    dx: float = 5.0,
    dy: float = 2.0,
    velocity: float = 1.0,
    depth: float = 2.0,
    shear: float = 10.0,
    wet_halfwidth: float | None = None,
    nz: int | None = None,
) -> tuple[str, str | None]:
    """Write a uniform straight channel as npz grids for analytical tests.

    The channel runs along +x from 0 to ``length`` and spans ``y`` in ``[-width/2, width/2]``.
    On wet nodes bed elevation is 0, water surface is ``depth``, velocity is ``(velocity, 0, 0)``
    and shear stress is ``shear`` (so u* = sqrt(shear / DEFAULT_WATER_DENSITY) with the
    ``shear_stress`` u* method), all uniform, which gives uniform diffusion coefficients and a
    solver whose exact answer is known.

    Args:
        dirpath: directory to write ``channel_2d.npz`` (and ``channel_3d.npz``) into
        length: channel length in x [m]
        width: full grid width in y [m]
        dx: node spacing in x [m]
        dy: node spacing in y [m]
        velocity: uniform x velocity [m/s]
        depth: uniform water depth [m]
        shear: uniform bed shear stress [Pa]
        wet_halfwidth: if given, nodes with ``|y| > wet_halfwidth`` are dry (``ibc = 0``; water surface,
            velocity and shear all zero there); otherwise the whole grid is wet
        nz: number of vertical levels for a 3D grid; ``None`` writes only the 2D grid. Cannot be
            combined with ``wet_halfwidth`` because dry nodes would give zero-thickness 3D cells.

    Raises:
        ValueError: if both ``nz`` and ``wet_halfwidth`` are given

    Returns:
        (path to 2D npz, path to 3D npz or None)
    """
    if nz is not None and wet_halfwidth is not None:
        raise ValueError("wet_halfwidth with nz would produce zero-thickness 3D cells on the dry margin")
    dirpath = pathlib.Path(dirpath)
    xs = np.arange(0.0, length + 0.5 * dx, dx)
    ys = np.arange(-0.5 * width, 0.5 * width + 0.5 * dy, dy)
    x, y = np.meshgrid(xs, ys)  # shape (nn, ns), i (along x) fastest when raveled
    wet = np.ones_like(x) if wet_halfwidth is None else (np.abs(y) <= wet_halfwidth).astype(float)
    elev = np.zeros_like(x)
    wse = elev + depth * wet
    vx = velocity * wet
    path2d = dirpath / "channel_2d.npz"
    np.savez(
        path2d,
        x=x,
        y=y,
        elev=elev,
        ibc=wet,
        shear=shear * wet,
        vx=vx,
        vy=np.zeros_like(x),
        wse=wse,
    )
    if nz is None:
        return str(path2d), None
    frac = np.linspace(0.0, 1.0, nz)[:, None, None]
    x3 = np.broadcast_to(x, (nz, *x.shape))
    y3 = np.broadcast_to(y, (nz, *y.shape))
    z3 = elev[None] + frac * (wse - elev)[None]
    vx3 = np.broadcast_to(vx, (nz, *vx.shape))
    path3d = dirpath / "channel_3d.npz"
    np.savez(path3d, x=x3, y=y3, z=z3, vx=vx3, vy=np.zeros_like(vx3), vz=np.zeros_like(vx3))
    return str(path2d), str(path3d)
