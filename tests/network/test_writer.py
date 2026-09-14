"""Tests for NetworkWriter round trips."""

import json

import numpy as np
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
