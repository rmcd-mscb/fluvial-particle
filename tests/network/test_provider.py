"""Tests for FileHydraulicsProvider: validation, static arrays, subsetting."""

import datetime as dt

import numpy as np
import pytest

from fluvial_particle.network.provider import FileHydraulicsProvider, HydraulicsProvider
from tests.network.support import ArrayHydraulicsProvider, chain_dataset, three_reach_dataset, write_network_file


@pytest.fixture
def three_file(tmp_path):
    return write_network_file(tmp_path / "three.nc", three_reach_dataset(temperature=4.0))


def test_providers_satisfy_hydraulics_provider_protocol(three_file):
    # HydraulicsProvider's data members are read-only properties in the Protocol so that a concrete
    # provider's plain (mutable) instance attributes still satisfy it structurally, without a cast.
    with FileHydraulicsProvider(three_file) as prov:
        assert isinstance(prov, HydraulicsProvider)
    array_prov = ArrayHydraulicsProvider(
        [np.datetime64("1979-01-01", "ns")],
        {"reach_id": np.array([101])},
        {name: np.array([1.0]) for name in ("flow_in", "flow_out", "velocity", "depth", "width", "ustar")},
    )
    assert isinstance(array_prov, HydraulicsProvider)


def test_open_static_and_times(three_file):
    with FileHydraulicsProvider(three_file) as prov:
        assert prov.n_reach == 3
        assert prov.times.dtype == np.dtype("datetime64[ns]")
        assert prov.times.size == 3
        assert list(prov.static["reach_id"]) == [101, 102, 103]
        assert list(prov.static["to_index"]) == [2, 2, -1]
        assert prov.static["length"].dtype == np.float64
        assert prov.crs_wkt == 'PROJCRS["test"]'
        assert prov.dtype == np.dtype("float64")
        assert prov.subset_index is None
        assert prov.has_temperature


def test_polylines_are_lazy(three_file):
    with FileHydraulicsProvider(three_file) as prov:
        assert "reach_vertex_start" in prov.static
        assert not prov.static.polylines_loaded
        assert prov.static["vertex_x"].shape == (8,)
        assert prov.static.polylines_loaded
        assert list(prov.static["reach_vertex_start"]) == [0, 2, 5]


def test_missing_required_variable_lists_all(tmp_path):
    ds = three_reach_dataset().drop_vars(["velocity", "ustar"])
    path = write_network_file(tmp_path / "bad.nc", ds)
    with pytest.raises(ValueError, match="velocity") as exc:
        FileHydraulicsProvider(path)
    assert "ustar" in str(exc.value)


def test_wrong_units_and_missing_units_both_raise(tmp_path):
    ds = three_reach_dataset()
    ds["velocity"].attrs["units"] = "ft s-1"
    with pytest.raises(ValueError, match=r"velocity.*units"):
        FileHydraulicsProvider(write_network_file(tmp_path / "u1.nc", ds))
    ds = three_reach_dataset()
    del ds["depth"].attrs["units"]
    with pytest.raises(ValueError, match=r"depth has no units"):
        FileHydraulicsProvider(write_network_file(tmp_path / "u2.nc", ds))


def test_negative_field_values_raise(tmp_path):
    ds = three_reach_dataset()
    ds["velocity"].values[1, 2] = -1.0
    with pytest.raises(ValueError, match=r"velocity has 1 negative value") as exc:
        FileHydraulicsProvider(write_network_file(tmp_path / "neg.nc", ds))
    assert "non-negative" in str(exc.value)


def test_non_finite_field_values_raise(tmp_path):
    ds = three_reach_dataset()
    ds["depth"].values[0, 0] = np.nan
    ds["depth"].values[2, 1] = np.inf
    with pytest.raises(ValueError, match=r"depth has 2 non-finite value"):
        FileHydraulicsProvider(write_network_file(tmp_path / "nan.nc", ds))


def test_bad_to_index_and_cycle(tmp_path):
    ds = three_reach_dataset(to_index=[2, 5, -1])
    with pytest.raises(ValueError, match="to_index"):
        FileHydraulicsProvider(write_network_file(tmp_path / "range.nc", ds))
    ds = three_reach_dataset(to_index=[2, 2, 0])  # 0 -> 2 -> 0 cycle
    with pytest.raises(ValueError, match="cycle"):
        FileHydraulicsProvider(write_network_file(tmp_path / "cycle.nc", ds))


def test_non_positive_length_raises(tmp_path):
    ds = three_reach_dataset(length=[1000.0, 0.0, 3000.0])
    with pytest.raises(ValueError, match="length") as exc:
        FileHydraulicsProvider(write_network_file(tmp_path / "len0.nc", ds))
    assert "1" in str(exc.value)


def test_is_outlet_inconsistent(tmp_path):
    ds = three_reach_dataset()
    ds["is_outlet"].values[:] = 0
    with pytest.raises(ValueError, match="is_outlet"):
        FileHydraulicsProvider(write_network_file(tmp_path / "outlet.nc", ds))


def test_polyline_block_validation(tmp_path):
    ds = three_reach_dataset()
    ds["reach_vertex_count"].values[2] = 10
    with pytest.raises(ValueError, match="reach_vertex"):
        FileHydraulicsProvider(write_network_file(tmp_path / "pl1.nc", ds)).static["vertex_x"]
    ds = three_reach_dataset()
    ds["vertex_dist"].values[6] = 5000.0  # not monotone inside reach 2
    with pytest.raises(ValueError, match="vertex_dist"):
        FileHydraulicsProvider(write_network_file(tmp_path / "pl2.nc", ds)).static["vertex_x"]


def test_time_not_increasing(tmp_path):
    t = np.array(["1979-01-02", "1979-01-01", "1979-01-03"], dtype="datetime64[ns]")
    ds = three_reach_dataset(times=t)
    with pytest.raises(ValueError, match="time"):
        FileHydraulicsProvider(write_network_file(tmp_path / "t.nc", ds))


def test_subset_by_ids_remaps_to_index(three_file):
    with FileHydraulicsProvider(three_file, reach_subset=[102, 101]) as prov:
        assert prov.n_reach == 2
        assert list(prov.static["reach_id"]) == [101, 102]  # file order kept
        assert list(prov.static["to_index"]) == [-1, -1]  # downstream reach dropped -> outlet
        assert list(prov.static["is_outlet"]) == [1, 1]
        assert list(prov.subset_index) == [0, 1]
        assert prov.static["vertex_x"].shape == (5,)
        assert list(prov.static["reach_vertex_start"]) == [0, 2]
    with pytest.raises(KeyError):
        FileHydraulicsProvider(three_file, reach_subset=[999])


def test_subset_by_outlet_closure(tmp_path):
    path = write_network_file(tmp_path / "chain.nc", chain_dataset(n_reach=6))
    with FileHydraulicsProvider(path, reach_subset={"outlet": 4}) as prov:
        assert list(prov.static["reach_id"]) == [1, 2, 3, 4]
        assert list(prov.static["to_index"]) == [1, 2, 3, -1]


def test_chunk_warning(tmp_path):
    ds = three_reach_dataset()
    path = tmp_path / "chunked.nc"
    ds.to_netcdf(path, engine="h5netcdf", encoding={"velocity": {"chunksizes": (3, 1)}})
    with pytest.warns(UserWarning, match="chunk"):
        FileHydraulicsProvider(path).close()


def test_memory_estimate(three_file):
    with FileHydraulicsProvider(three_file, dtype="float32") as prov:
        est = prov.memory_estimate(1000)
        assert est["window_bytes"] == 2 * 3 * 7 * 4  # 2 slices, 3 reaches, 6 fields + temperature, float32
        assert est["particle_bytes"] > 0
        assert est["static_bytes"] > 0


def _stepped_file(tmp_path, n_time=4):
    n = 3
    vel = np.arange(1, n_time + 1, dtype=float)[:, None] * np.ones((1, n))  # day k has velocity k+1
    flow = np.full((n_time, n), 10.0)
    t = np.datetime64("1979-01-01", "ns") + np.arange(n_time) * np.timedelta64(1, "D")
    return write_network_file(tmp_path / "step.nc", three_reach_dataset(velocity=vel, flow_out=flow, times=t))


def test_hold_and_linear(tmp_path):
    path = _stepped_file(tmp_path)
    noon = np.datetime64("1979-01-02T12:00", "ns")
    with FileHydraulicsProvider(path, interpolation="hold") as prov:
        assert prov.hydraulics(noon)["velocity"][0] == 2.0
        assert prov.hydraulics(np.datetime64("1979-01-04", "ns"))["velocity"][0] == 4.0
    with FileHydraulicsProvider(path, interpolation="linear") as prov:
        assert prov.hydraulics(noon)["velocity"][0] == pytest.approx(2.5)
        assert prov.hydraulics(np.datetime64("1979-01-04", "ns"))["velocity"][0] == 4.0
        h = prov.hydraulics(np.datetime64("1979-01-04", "ns"))
        assert set(h) == {"flow_in", "flow_out", "velocity", "depth", "width", "ustar"}
        assert h["velocity"].dtype == np.float64


def test_linear_zero_flow_guard(tmp_path):
    flow = np.array([[10.0, 0.0, 10.0], [10.0, 10.0, 0.0]])
    vel = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    t = np.datetime64("1979-01-01", "ns") + np.arange(2) * np.timedelta64(1, "D")
    path = write_network_file(tmp_path / "dry.nc", three_reach_dataset(velocity=vel, flow_out=flow, times=t))
    with FileHydraulicsProvider(path) as prov:
        h = prov.hydraulics(np.datetime64("1979-01-01T12:00", "ns"))
        assert h["velocity"][0] == 1.0
        assert h["velocity"][1] == 0.0 and h["ustar"][1] == 0.0
        assert h["velocity"][2] == 0.0 and h["ustar"][2] == 0.0
        assert h["flow_out"][1] == pytest.approx(5.0)  # flow itself still interpolates


def test_window_advances_one_read_at_a_time(tmp_path, monkeypatch):
    path = _stepped_file(tmp_path, n_time=5)
    with FileHydraulicsProvider(path) as prov:
        reads = []
        orig = prov._read_slice
        monkeypatch.setattr(prov, "_read_slice", lambda k: (reads.append(k), orig(k))[1])
        day = np.timedelta64(1, "D")
        t0 = np.datetime64("1979-01-01", "ns")
        prov.hydraulics(t0 + np.timedelta64(6, "h"))
        assert reads == [0, 1]
        assert prov.window_indices == (0, 1)
        prov.hydraulics(t0 + np.timedelta64(18, "h"))
        assert reads == [0, 1]
        prov.hydraulics(t0 + day + np.timedelta64(1, "h"))
        assert reads == [0, 1, 2]
        assert prov.window_indices == (1, 2)
        prov.hydraulics(t0 + 3 * day + np.timedelta64(1, "h"))  # skip ahead: two reads
        assert reads == [0, 1, 2, 3, 4]
        prov.hydraulics(t0 + np.timedelta64(1, "h"))  # backwards: reset, two reads
        assert reads == [0, 1, 2, 3, 4, 0, 1]


def test_out_of_range_and_time_window(tmp_path):
    path = _stepped_file(tmp_path)
    with FileHydraulicsProvider(path) as prov:
        with pytest.raises(ValueError, match="outside"):
            prov.hydraulics(np.datetime64("1978-12-31", "ns"))
        prov.time_window = (np.datetime64("1979-01-02", "ns"), np.datetime64("1979-01-03", "ns"))
        with pytest.raises(ValueError, match="outside"):
            prov.hydraulics(np.datetime64("1979-01-03T01:00", "ns"))
        with pytest.raises(ValueError, match="time_window"):
            prov.time_window = (np.datetime64("1979-01-03", "ns"), np.datetime64("1979-01-02", "ns"))


def test_subset_and_dtype_apply_to_fields(tmp_path):
    path = _stepped_file(tmp_path)
    with FileHydraulicsProvider(path, reach_subset=[103], dtype="float32") as prov:
        h = prov.hydraulics(np.datetime64("1979-01-01", "ns"))
        assert h["velocity"].shape == (1,)
        assert h["velocity"].dtype == np.float32


def test_single_timestamp_file(tmp_path):
    t = np.array(["1979-01-01"], dtype="datetime64[ns]")
    path = write_network_file(tmp_path / "one.nc", three_reach_dataset(times=t))
    with FileHydraulicsProvider(path) as prov:
        assert prov.hydraulics(t[0])["velocity"][0] == 1.0


def test_hydraulics_accepts_strings_and_datetimes(tmp_path):
    path = _stepped_file(tmp_path)
    noon = np.datetime64("1979-01-02T12:00", "ns")
    with FileHydraulicsProvider(path, interpolation="linear") as prov:
        expected = prov.hydraulics(noon)["velocity"][0]
        assert prov.hydraulics("1979-01-02T12:00")["velocity"][0] == expected
        assert prov.hydraulics(dt.datetime(1979, 1, 2, 12))["velocity"][0] == expected
