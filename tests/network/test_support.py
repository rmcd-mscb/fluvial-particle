"""Tests for the network test-support builders."""

import numpy as np
import xarray as xr

from tests.network.support import ArrayHydraulicsProvider, chain_dataset, three_reach_dataset, write_network_file


def test_three_reach_dataset_schema():
    ds = three_reach_dataset()
    assert ds.sizes == {"reach": 3, "time": 3, "vertex": 8}
    for name in ("flow_in", "flow_out", "velocity", "depth", "width", "ustar"):
        assert ds[name].dims == ("time", "reach")
        assert "units" in ds[name].attrs
    assert list(ds["to_index"].values) == [2, 2, -1]
    assert list(ds["is_outlet"].values) == [0, 0, 1]
    assert list(ds["reach_vertex_start"].values) == [0, 2, 5]
    assert list(ds["reach_vertex_count"].values) == [2, 3, 3]
    np.testing.assert_allclose(ds["vertex_dist"].values[5:8], [0.0, 1500.0, 3000.0])
    np.testing.assert_allclose(ds["ustar"].values[0], np.sqrt(9.80665 * np.array([1.0, 1.0, 2.0]) * 0.001))


def test_write_and_reopen(tmp_path):
    path = write_network_file(tmp_path / "net.nc", three_reach_dataset(temperature=5.0))
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds["time"].dtype.kind == "M"
        assert "water_temperature" in ds
        assert ds.attrs["conventions_note"]


def test_chain_dataset_hits_target_dispersion():
    ds = chain_dataset(n_reach=4, length=500.0, velocity=1.0, k_target=10.0)
    v, d, w, u = (ds[n].values[0] for n in ("velocity", "depth", "width", "ustar"))
    np.testing.assert_allclose(0.011 * v**2 * w**2 / (d * u), 10.0)
    assert list(ds["to_index"].values) == [1, 2, 3, -1]


def test_array_provider_hold_semantics():
    ds = three_reach_dataset(velocity=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
    prov = ArrayHydraulicsProvider.from_dataset(ds)
    t = np.datetime64("1979-01-02T12:00", "ns")
    assert prov.hydraulics(t)["velocity"][0] == 2.0
    assert prov.n_reach == 3
    assert "reach_vertex_start" in prov.static
