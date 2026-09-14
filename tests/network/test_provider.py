"""Tests for FileHydraulicsProvider: validation, static arrays, subsetting."""

import numpy as np
import pytest

from fluvial_particle.network.provider import FileHydraulicsProvider
from tests.network.support import chain_dataset, three_reach_dataset, write_network_file


@pytest.fixture
def three_file(tmp_path):
    return write_network_file(tmp_path / "three.nc", three_reach_dataset(temperature=4.0))


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


def test_wrong_units_raise_and_missing_units_warn(tmp_path):
    ds = three_reach_dataset()
    ds["velocity"].attrs["units"] = "ft s-1"
    with pytest.raises(ValueError, match=r"velocity.*units"):
        FileHydraulicsProvider(write_network_file(tmp_path / "u1.nc", ds))
    ds = three_reach_dataset()
    del ds["depth"].attrs["units"]
    with pytest.warns(UserWarning, match="depth"):
        FileHydraulicsProvider(write_network_file(tmp_path / "u2.nc", ds)).close()


def test_bad_to_index_and_cycle(tmp_path):
    ds = three_reach_dataset(to_index=[2, 5, -1])
    with pytest.raises(ValueError, match="to_index"):
        FileHydraulicsProvider(write_network_file(tmp_path / "range.nc", ds))
    ds = three_reach_dataset(to_index=[2, 2, 0])  # 0 -> 2 -> 0 cycle
    with pytest.raises(ValueError, match="cycle"):
        FileHydraulicsProvider(write_network_file(tmp_path / "cycle.nc", ds))


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
