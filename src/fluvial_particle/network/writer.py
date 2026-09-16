"""NetCDF4 particle output through h5netcdf on h5py, serial or with the mpio driver."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import h5netcdf
import numpy as np
import numpy.typing as npt

from .particles import StateVar
from .sources import ParticleSchedule


OUTPUT_FILENAME = "network_particles.nc"
MAX_PARTICLE_CHUNK = 2**18


class NetworkWriter:
    """Write the network particle file one output time at a time.

    Every rank constructs the writer (dataset creation is collective under mpio); each rank writes its own
    particle slice ``lo:hi``. The ``time`` dimension is unlimited and resized collectively in write_step.

    Args:
        path: output file.
        n_particles: total particle count across ranks.
        reach_id: ids of the run's reaches (subset order).
        start_time: run start; ``time`` is stored as seconds since it.
        attrs: global attributes (strings, numbers, or None; None is skipped).
        dtype: dtype of ``s``.
        comm: MPI communicator for parallel writes, or None.
        state_specs: the particle model's declared state; one variable per spec with ``output=True``.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        n_particles: int,
        reach_id: npt.NDArray[np.integer[Any]],
        start_time: np.datetime64,
        attrs: Mapping[str, Any],
        dtype: npt.DTypeLike = np.float64,
        comm: Any = None,
        state_specs: Sequence[StateVar] = (),
    ) -> None:
        """Create the output file and all variables (collective under mpio)."""
        self.path = pathlib.Path(path)
        self._rank = int(comm.Get_rank()) if comm is not None else 0
        kwargs: dict[str, Any] = {"driver": "mpio", "comm": comm} if comm is not None else {}
        self._f = h5netcdf.File(str(self.path), "w", **kwargs)
        n = int(n_particles)
        self._f.dimensions = {"time": None, "particle": n, "reach": int(reach_id.size)}
        chunk = (1, max(1, min(n, MAX_PARTICLE_CHUNK)))
        f = self._f
        start_iso = str(start_time.astype("datetime64[s]"))
        v = f.create_variable("time", ("time",), dtype="f8", chunks=(1024,))
        v.attrs["units"] = f"seconds since {start_iso}"
        v.attrs["calendar"] = "proleptic_gregorian"
        v.attrs["long_name"] = "output time"
        v = f.create_variable("time_seconds", ("time",), dtype="f8", chunks=(1024,))
        v.attrs["units"] = "s"
        v.attrs["long_name"] = "seconds since start_time"
        v = f.create_variable("reach_index", ("time", "particle"), dtype="i4", chunks=chunk, fillvalue=-1)
        v.attrs["long_name"] = "reach index of the particle, -1 when not active"
        # -1 is a real, in-range value here (not a "missing data" marker), so drop the CF _FillValue
        # attribute h5netcdf wrote from fillvalue=-1: otherwise xarray's default CF decoding would mask
        # every -1 to NaN and upcast the variable to float64.
        del v.attrs["_FillValue"]
        v = f.create_variable("s", ("time", "particle"), dtype=np.dtype(dtype), chunks=chunk, fillvalue=np.nan)
        v.attrs["units"] = "m"
        v.attrs["long_name"] = "distance from the reach's upstream end, NaN when not active"
        v = f.create_variable("status", ("time", "particle"), dtype="i1", chunks=chunk, fillvalue=0)
        v.attrs["long_name"] = "particle status"
        v.attrs["flag_values"] = np.array([0, 1, 2, 3, 4], dtype=np.int8)
        v.attrs["flag_meanings"] = "unreleased active exited settled removed"
        # Same issue as reach_index: 0 ("unreleased") is a real status value, not a fill sentinel.
        del v.attrs["_FillValue"]
        for name, dt_, units, long_name, fill in (
            ("mass", "f8", None, "particle mass", None),
            ("source_index", "i4", None, "row in the source table", None),
            ("release_reach", "i4", None, "release reach index", None),
            ("release_s", "f8", "m", "release distance from the reach's upstream end", None),
            ("release_time", "f8", "s", "release time, seconds since start_time", None),
            (
                "exit_time",
                "f8",
                "s",
                "time the particle reached a terminal status, seconds since start_time; NaN until then",
                np.nan,
            ),
            (
                "exit_reach",
                "i4",
                None,
                "reach index at the terminal status: the outlet for exited, the bed reach for settled; -1 until then",
                -1,
            ),
        ):
            var_kwargs: dict[str, Any] = {} if fill is None else {"fillvalue": fill}
            v = f.create_variable(name, ("particle",), dtype=dt_, **var_kwargs)
            if units:
                v.attrs["units"] = units
            v.attrs["long_name"] = long_name
            if name == "exit_reach":
                # -1 ("not yet exited") is a real value here too; see reach_index above.
                del v.attrs["_FillValue"]
        v = f.create_variable("reach_id", ("reach",), dtype="i8", data=np.asarray(reach_id, dtype=np.int64))
        v.attrs["long_name"] = "reach ids in the run's reach order"
        self._state_specs = tuple(spec for spec in state_specs if spec.output)
        for spec in self._state_specs:
            self._create_state_variable(spec, chunk)
        f.attrs["start_time"] = start_iso
        for key, value in attrs.items():
            if value is not None:
                f.attrs[key] = value
        self._n_time = 0

    def _create_state_variable(self, spec: StateVar, chunk: tuple[int, int]) -> None:
        """Create the output variable (and vector dimension with its labels) for one declared state."""
        f = self._f
        dims: tuple[str, ...] = ("time", "particle")
        chunks: tuple[int, ...] = chunk
        if spec.shape:
            assert spec.dim is not None  # StateVar validates this
            if spec.dim not in f.dimensions:
                f.dimensions[spec.dim] = int(spec.shape[0])
                if spec.labels is not None:
                    # Fixed-width bytes rather than variable-length strings: vlen types cannot be
                    # written under the mpio driver. xarray decodes them back to str via _Encoding.
                    labels = np.asarray(spec.labels).astype("S")
                    lv = f.create_variable(spec.dim, (spec.dim,), dtype=labels.dtype, data=labels)
                    lv.attrs["_Encoding"] = "utf-8"
            dims = (*dims, spec.dim)
            chunks = (*chunk, int(spec.shape[0]))
        dtype = np.dtype(spec.dtype)
        v = f.create_variable(spec.name, dims, dtype=dtype, chunks=chunks, fillvalue=dtype.type(spec.fill))
        if dtype.kind != "f":
            # An integer fill is a real value (see reach_index); keep xarray from masking it to NaN.
            del v.attrs["_FillValue"]
        if spec.units:
            v.attrs["units"] = spec.units
        if spec.long_name:
            v.attrs["long_name"] = spec.long_name
        if spec.kind is not None:
            v.attrs["kind"] = spec.kind
        v.attrs["fluvial_particle_state"] = np.int8(1)

    def __enter__(self) -> NetworkWriter:
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the file on context exit."""
        self.close()

    def write_schedule(self, schedule: ParticleSchedule, lo: int, hi: int) -> None:
        """Write the per-particle release arrays for particles lo..hi-1.

        Args:
            schedule: the release schedule slice to write.
            lo: first particle index (inclusive).
            hi: last particle index (exclusive).
        """
        f = self._f
        f.variables["mass"][lo:hi] = schedule.mass
        f.variables["source_index"][lo:hi] = schedule.source_index
        f.variables["release_reach"][lo:hi] = schedule.release_reach
        f.variables["release_s"][lo:hi] = schedule.release_s
        f.variables["release_time"][lo:hi] = schedule.release_time

    def write_step(
        self,
        itime: int,
        time_seconds: float,
        reach: npt.NDArray[np.integer[Any]],
        s: npt.NDArray[np.floating[Any]],
        status: npt.NDArray[np.integer[Any]],
        lo: int,
        hi: int,
        state: Mapping[str, npt.NDArray[Any]] | None = None,
    ) -> None:
        """Write one output time for particles lo..hi-1 (resizes ``time`` when itime is new; collective).

        Args:
            itime: output time index.
            time_seconds: seconds since start_time for this output time.
            reach: reach index per particle, -1 when not active.
            s: distance from the reach's upstream end per particle, NaN when not active.
            status: particle status per particle (0 unreleased, 1 active, 2 exited, 3 settled, 4 removed).
            lo: first particle index (inclusive).
            hi: last particle index (exclusive).
            state: the model's declared state arrays for these particles (``solver.state``); every
                spec passed as ``state_specs`` with ``output=True`` must be present.
        """
        f = self._f
        if itime >= self._n_time:
            # Collective on all ranks under mpio: every rank must call resize_dimension in lockstep.
            f.resize_dimension("time", itime + 1)
            self._n_time = itime + 1
        if self._rank == 0:
            # Under mpio every rank sees the same (itime, time_seconds); write it once from rank 0 so
            # ranks don't race on the same element.
            f.variables["time"][itime] = float(time_seconds)
            f.variables["time_seconds"][itime] = float(time_seconds)
        f.variables["reach_index"][itime, lo:hi] = np.asarray(reach, dtype=np.int32)
        f.variables["s"][itime, lo:hi] = s
        f.variables["status"][itime, lo:hi] = np.asarray(status, dtype=np.int8)
        for spec in self._state_specs:
            if state is None or spec.name not in state:
                raise ValueError(f"write_step needs the declared state {spec.name!r}")
            f.variables[spec.name][itime, lo:hi] = np.asarray(state[spec.name], dtype=np.dtype(spec.dtype))

    def write_exits(
        self, exit_time: npt.NDArray[np.floating[Any]], exit_reach: npt.NDArray[np.integer[Any]], lo: int, hi: int
    ) -> None:
        """Write terminal times and reaches for particles lo..hi-1 (called once at the end of the run).

        Args:
            exit_time: seconds since start_time at the terminal status per particle, NaN if still active.
            exit_reach: reach index at the terminal status per particle, -1 if still active.
            lo: first particle index (inclusive).
            hi: last particle index (exclusive).
        """
        self._f.variables["exit_time"][lo:hi] = exit_time
        self._f.variables["exit_reach"][lo:hi] = np.asarray(exit_reach, dtype=np.int32)

    def close(self) -> None:
        """Flush and close the file."""
        self._f.close()
