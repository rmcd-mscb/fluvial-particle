"""Tests for NetworkWriter round trips."""

import json

import numpy as np
import pytest
import xarray as xr

from fluvial_particle.network.sources import ParticleSchedule
from fluvial_particle.network.writer import OUTPUT_FILENAME, NetworkWriter


T0 = np.datetime64("1979-01-01", "ns")


def test_round_trip(tmp_path):
    path = tmp_path / OUTPUT_FILENAME
    sch = ParticleSchedule(
        np.array([0, 0, 1], dtype=np.int32),
        np.array([0.0, 5.0, 10.0]),
        np.array([0.0, 0.0, 30.0]),
        np.array([1.0, 1.0, 2.0]),
        np.array([0, 0, 1], dtype=np.int32),
    )
    attrs = {"dt": 60.0, "sources": json.dumps([{"reach_id": 1}]), "mass_units": "kg"}
    with NetworkWriter(path, n_particles=3, reach_id=np.array([101, 102]), start_time=T0, attrs=attrs) as w:
        w.write_schedule(sch, 0, 3)
        w.write_step(
            0,
            0.0,
            np.array([0, 0, -1], dtype=np.int32),
            np.array([0.0, 5.0, np.nan]),
            np.array([1, 1, 0], dtype=np.int8),
            0,
            3,
        )
        w.write_step(
            1,
            60.0,
            np.array([-1, 0, 1], dtype=np.int32),
            np.array([np.nan, 65.0, 40.0]),
            np.array([2, 1, 1], dtype=np.int8),
            0,
            3,
        )
        w.write_exits(np.array([42.0, np.nan, np.nan]), np.array([1, -1, -1], dtype=np.int32), 0, 3)
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds.sizes == {"time": 2, "particle": 3, "reach": 2}
        assert ds["time"].values[1] == T0 + np.timedelta64(60, "s")
        assert list(ds["time_seconds"].values) == [0.0, 60.0]
        assert ds["reach_index"].dtype == np.int32 and list(ds["reach_index"].values[1]) == [-1, 0, 1]
        assert np.isnan(ds["s"].values[0, 2]) and ds["s"].values[1, 1] == 65.0
        assert list(ds["status"].values[0]) == [1, 1, 0]
        assert list(ds["mass"].values) == [1.0, 1.0, 2.0]
        assert list(ds["release_time"].values) == [0.0, 0.0, 30.0]
        assert ds["exit_time"].values[0] == 42.0 and np.isnan(ds["exit_time"].values[1])
        assert list(ds["exit_reach"].values) == [1, -1, -1]
        assert list(ds["reach_id"].values) == [101, 102]
        assert ds.attrs["dt"] == 60.0 and json.loads(ds.attrs["sources"])[0]["reach_id"] == 1
        assert ds["reach_index"].encoding.get("chunksizes") == (1, 3)


def test_serial_writer_has_rank_zero_and_writes_time(tmp_path):
    path = tmp_path / OUTPUT_FILENAME
    sch = ParticleSchedule.simple(reach=0, s=0.0, time=0.0)
    with NetworkWriter(path, n_particles=1, reach_id=np.array([101]), start_time=T0, attrs={}, comm=None) as w:
        assert w._rank == 0
        w.write_schedule(sch, 0, 1)
        w.write_step(0, 0.0, np.array([0], dtype=np.int32), np.array([0.0]), np.array([1], dtype=np.int8), 0, 1)
        w.write_step(1, 60.0, np.array([0], dtype=np.int32), np.array([1.0]), np.array([1], dtype=np.int8), 0, 1)
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert list(ds["time_seconds"].values) == [0.0, 60.0]
        assert ds["time"].values[1] == T0 + np.timedelta64(60, "s")


def test_partial_slices_and_dtype(tmp_path):
    path = tmp_path / OUTPUT_FILENAME
    sch = ParticleSchedule(
        np.zeros(4, dtype=np.int32), np.zeros(4), np.zeros(4), np.ones(4), np.zeros(4, dtype=np.int32)
    )
    with NetworkWriter(path, n_particles=4, reach_id=np.array([1]), start_time=T0, attrs={}, dtype=np.float32) as w:
        w.write_schedule(sch.slice(0, 2), 0, 2)
        w.write_schedule(sch.slice(2, 4), 2, 4)
        w.write_step(
            0, 0.0, np.zeros(2, dtype=np.int32), np.full(2, 1.5, dtype=np.float32), np.ones(2, dtype=np.int8), 0, 2
        )
        w.write_step(
            0, 0.0, np.zeros(2, dtype=np.int32), np.full(2, 2.5, dtype=np.float32), np.ones(2, dtype=np.int8), 2, 4
        )
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds["s"].dtype == np.float32
        assert list(ds["s"].values[0]) == [1.5, 1.5, 2.5, 2.5]


def test_declared_state_round_trip(tmp_path):
    from fluvial_particle.network.particles import StateVar

    specs = (
        StateVar("zeta", units="1", long_name="relative elevation"),
        StateVar("c", shape=(2,), dim="constituent", labels=("a", "b"), kind="extensive", units="kg"),
        StateVar("hidden", output=False),
        StateVar("stage", dtype="i2", fill=-1, long_name="life stage"),
    )
    state = {
        "zeta": np.array([0.1, 0.2, np.nan]),
        "c": np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]]),
        "hidden": np.zeros(3),
        "stage": np.array([0, 1, -1], dtype=np.int16),
    }
    path = tmp_path / OUTPUT_FILENAME
    sch = ParticleSchedule(
        np.zeros(3, dtype=np.int32), np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3, dtype=np.int32)
    )
    with NetworkWriter(
        path, n_particles=3, reach_id=np.array([101]), start_time=T0, attrs={"dt": 60.0}, state_specs=specs
    ) as w:
        w.write_schedule(sch, 0, 3)
        reach = np.array([0, 0, -1], dtype=np.int32)
        s = np.array([0.0, 5.0, np.nan])
        status = np.array([1, 1, 0], dtype=np.int8)
        w.write_step(0, 0.0, reach, s, status, 0, 3, state=state)
        state["zeta"][0] = 0.3
        w.write_step(1, 60.0, reach, s, status, 0, 3, state=state)
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds["zeta"].dims == ("time", "particle")
        assert ds["zeta"].attrs["units"] == "1" and ds["zeta"].attrs["long_name"] == "relative elevation"
        assert ds["zeta"].attrs["fluvial_particle_state"] == 1 and "kind" not in ds["zeta"].attrs
        np.testing.assert_allclose(ds["zeta"].values, [[0.1, 0.2, np.nan], [0.3, 0.2, np.nan]])
        assert ds["c"].dims == ("time", "particle", "constituent") and ds["c"].attrs["kind"] == "extensive"
        assert list(ds["constituent"].values) == ["a", "b"]
        np.testing.assert_allclose(ds["c"].values[0], [[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
        assert "hidden" not in ds
        assert ds["stage"].dtype == np.int16 and list(ds["stage"].values[1]) == [0, 1, -1]
        assert ds["reach_index"].attrs.get("fluvial_particle_state") is None


def test_write_step_requires_every_declared_state(tmp_path):
    from fluvial_particle.network.particles import StateVar

    path = tmp_path / OUTPUT_FILENAME
    with NetworkWriter(
        path, n_particles=2, reach_id=np.array([101]), start_time=T0, attrs={}, state_specs=(StateVar("zeta"),)
    ) as w:
        with pytest.raises(ValueError, match="zeta"):
            w.write_step(0, 0.0, np.zeros(2, dtype=np.int32), np.zeros(2), np.ones(2, dtype=np.int8), 0, 2, state={})
        # an MPI-style partial slice of a vector state
        w2 = w
        w2.write_step(
            0,
            0.0,
            np.zeros(1, dtype=np.int32),
            np.zeros(1),
            np.ones(1, dtype=np.int8),
            1,
            2,
            state={"zeta": np.array([0.3])},
        )
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds["zeta"].values[0, 1] == 0.3 and np.isnan(ds["zeta"].values[0, 0])
