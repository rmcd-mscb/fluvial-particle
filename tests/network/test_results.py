"""Tests for NetworkResults post-processing."""

import numpy as np
import pandas as pd
import pytest
import vtk
from vtk.util import numpy_support

from fluvial_particle.network.run import run_network_simulation
from tests.network.support import three_reach_dataset, write_network_file


T0 = np.datetime64("1979-01-01", "ns")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("res")
    path = write_network_file(tmp / "net.nc", three_reach_dataset())
    cfg = {
        "hydraulics_file": str(path),
        "dt": 600.0,
        "output_interval": 600.0,
        "end_time": "1979-01-01T02:00",
        "dispersion": {"model": "none"},
        "mass_units": "g",
        "sources": [
            {"reach_id": 101, "form": "slug", "time": 0.0, "mass": 3.0, "particles": 3},
            {"reach_id": 102, "form": "slug", "time": 1800.0, "mass": 1.0, "particles": 1, "s_frac": 0.5},
        ],
    }
    res = run_network_simulation(cfg, tmp / "out", seed=1, quiet=True)
    yield res
    res.close()


def test_basics(run):
    assert run.n_particles == 4 and run.n_reach == 3
    assert run.start_time == np.datetime64("1979-01-01", "ns")
    assert isinstance(run.sources, pd.DataFrame) and len(run.sources) == 2
    assert run.time_index(np.datetime64("1979-01-01T00:10", "ns")) == 1
    assert run.time_index(-1) == 12
    assert list(run.time_indices([0, 2])) == [0, 2]
    assert list(run.time_indices(slice(0, 2))) == [0, 1]
    assert run.time_indices(None).size == 13
    assert "particles" in run.summary() and "NetworkResults" in repr(run)


def test_time_index_out_of_range_raises(run):
    with pytest.raises(IndexError):
        run.positions(100)
    with pytest.raises(IndexError):
        run.positions(-100)


def test_positions_and_map(run):
    df = run.positions(1)  # t = 600 s
    assert list(df.columns) == ["particle", "reach_index", "reach_id", "s", "status", "mass"]
    assert df.loc[0, "reach_id"] == 101 and df.loc[0, "s"] == 600.0 and df.loc[3, "status"] == 0
    mp = run.map_positions(1)
    # reach 0 polyline (-1000,500)->(0,0), hydraulic 1000 m: s=600 -> 60% along
    assert mp.loc[0, "x"] == pytest.approx(-400.0) and mp.loc[0, "y"] == pytest.approx(200.0)
    assert np.isnan(mp.loc[3, "x"])
    assert len(run.polylines()) == 3
    ds = run.positions()
    assert set(ds.data_vars) >= {"reach_index", "s", "status"}


def test_arrivals(run):
    df = run.arrival_times()
    assert len(df) == 4  # all exit within 2 h
    np.testing.assert_allclose(df.loc[df.particle < 3, "exit_time"], 2500.0)
    # reach 102: 1000 m left at 0.5 m/s = 2000 s, then 3000 m at 2 m/s = 1500 s, released at 1800
    assert df.loc[df.particle == 3, "exit_time"].item() == pytest.approx(1800.0 + 3500.0)
    assert set(df.columns) >= {
        "particle",
        "exit_time",
        "exit_datetime",
        "exit_reach",
        "exit_reach_id",
        "release_reach_id",
        "mass",
    }
    assert len(run.arrival_times(outlet=103)) == 4 and len(run.arrival_times(outlet=101)) == 0
    hist = run.arrival_histogram(103, bin_seconds=3600.0)
    assert list(hist.columns) == ["time_start", "time", "mass"]
    assert hist["mass"].sum() == pytest.approx(4.0)
    assert hist.loc[0, "mass"] == pytest.approx(3.0)


def test_to_dataframe(run):
    long = run.to_dataframe(1)
    assert len(long) == 4 and "time" in long.columns
    assert len(run.to_dataframe()) == 4 * 13


def test_counts_and_concentration(run):
    c = run.counts(1, bin_length=100.0)  # t = 600 s: three particles at s = 600 in reach 0 (width 10, depth 1)
    assert c.dims == ("bin",) and c.sum().item() == 3
    assert c.values[6] == 3
    conc = run.concentration(1, bin_length=100.0)
    assert conc.attrs["units"] == "g m-3"
    assert conc.values[6] == pytest.approx(3.0 / (10.0 * 1.0 * 100.0))
    assert conc.coords["reach_id"].values[6] == 101 and conc.coords["s_start"].values[6] == 600.0
    assert np.nansum(conc.values[:10]) == pytest.approx(conc.values[6])
    rc = run.reach_concentration(1)
    assert rc.sizes["bin"] == 3 and rc.values[0] == pytest.approx(3.0 / (10.0 * 1.0 * 1000.0))
    with pytest.raises(TypeError):  # smoothing is gone: one bin per reach has nothing to smooth
        run.reach_concentration(1, smoothing=100.0)


def test_concentration_multi_time_and_smoothing(run):
    cube = run.concentration([0, 1, 2], bin_length=100.0)
    assert cube.dims == ("time", "bin") and cube.sizes["time"] == 3
    sm = run.concentration(1, bin_length=100.0, smoothing=150.0)
    bins = run.bins(100.0)
    vol = bins.bin_width * 10.0 * 1.0
    reach0 = bins.bin_reach == 0
    assert np.nansum(sm.values[reach0] * vol[reach0]) == pytest.approx(3.0)  # mass conserved in the reach
    assert (sm.values[reach0] > 0).sum() > 1  # spread over neighbors
    auto = run.concentration(1, bin_length=100.0, smoothing="auto")  # K = 0 -> tiny bandwidth -> same as binned
    assert np.nansum(auto.values[reach0] * vol[reach0]) == pytest.approx(3.0)


def test_dry_reach_is_nan(tmp_path):
    ds = three_reach_dataset(velocity=[1.0, 0.0, 2.0], flow_out=[10.0, 0.0, 80.0])
    path = write_network_file(tmp_path / "dry.nc", ds)
    cfg = {
        "hydraulics_file": str(path),
        "dt": 600.0,
        "output_interval": 600.0,
        "end_time": "1979-01-01T00:20",
        "dispersion": {"model": "none"},
        "sources": [{"reach_id": 102, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 1}],
    }
    with run_network_simulation(cfg, tmp_path / "out", seed=1, quiet=True) as res:
        conc = res.reach_concentration(1)
        assert np.isnan(conc.values[1]) and res.counts(1, np.inf).values[1] == 1


def test_concentration_requires_mass_units(run, tmp_path):
    # A foreign file (or one written by an older version) must fail loudly rather than report
    # concentrations labelled with a default unit the run never used.
    import shutil

    import h5py

    from fluvial_particle.network.results import NetworkResults

    copy = tmp_path / "no_units.nc"
    shutil.copy(run.path, copy)
    with h5py.File(copy, "r+") as f:
        del f.attrs["mass_units"]
    with NetworkResults(copy) as res:
        with pytest.raises(KeyError, match="mass_units"):
            res.concentration(1, bin_length=100.0)
        assert "mass units ?" in res.summary()  # summary() stays tolerant


def test_provider_reopens_with_the_run_dtype(tmp_path):
    path = write_network_file(tmp_path / "f32.nc", three_reach_dataset())
    cfg = {
        "hydraulics_file": str(path),
        "dtype": "float32",
        "dt": 600.0,
        "output_interval": 600.0,
        "end_time": "1979-01-01T00:20",
        "dispersion": {"model": "none"},
        "sources": [{"reach_id": 101, "form": "slug", "time": 0.0, "mass": 1.0, "particles": 1}],
    }
    with run_network_simulation(cfg, tmp_path / "out", seed=1, quiet=True) as res:
        assert res.attrs["dtype"] == "float32"
        assert res.provider.dtype == np.dtype("float32")
        assert res.provider.hydraulics(res.times[0])["velocity"].dtype == np.float32


def test_persist(run, tmp_path):
    out = run.persist(tmp_path / "conc.nc", bin_length=500.0)
    import xarray as xr

    with xr.open_dataset(out, engine="h5netcdf") as ds:
        assert ds["concentration"].dims == ("time", "bin") and ds.sizes["time"] == 13


def test_to_vtp(run, tmp_path):
    pvd = run.to_vtp(tmp_path / "vtk", times=[1, 2])
    assert pvd.name == "network.pvd" and pvd.exists()
    assert sorted(p.name for p in (tmp_path / "vtk" / "vtp").glob("*.vtp")) == ["network_0001.vtp", "network_0002.vtp"]

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(tmp_path / "vtk" / "vtp" / "network_0001.vtp"))
    reader.Update()
    polydata = reader.GetOutput()

    active = run.positions(1)
    active = active[active["status"] == 1]

    assert polydata.GetNumberOfPoints() == len(active)
    point_data = polydata.GetPointData()
    for name in ["reach_index", "s", "mass", "status", "source_index"]:
        assert point_data.GetArray(name) is not None

    reach_index = numpy_support.vtk_to_numpy(point_data.GetArray("reach_index"))
    status = numpy_support.vtk_to_numpy(point_data.GetArray("status"))
    assert reach_index.dtype.kind == "i"
    assert status.dtype.kind == "i"

    s = numpy_support.vtk_to_numpy(point_data.GetArray("s"))
    np.testing.assert_allclose(s, active["s"].to_numpy())


def test_to_vtp_writes_particles_in_a_single_vertex_reach(tmp_path):
    # Reach 101's polyline has one vertex: its particles map to that vertex instead of being
    # dropped from the VTP file for want of coordinates.
    ds = three_reach_dataset(
        polylines=[
            [(-1000.0, 500.0)],
            [(-2000.0, -500.0), (-1000.0, -250.0), (0.0, 0.0)],
            [(0.0, 0.0), (1500.0, 0.0), (3000.0, 0.0)],
        ]
    )
    path = write_network_file(tmp_path / "degenerate.nc", ds)
    cfg = {
        "hydraulics_file": str(path),
        "dt": 300.0,
        "output_interval": 300.0,
        "end_time": "1979-01-01T00:10",
        "dispersion": {"model": "none"},
        "sources": [{"reach_id": 101, "form": "slug", "time": 0.0, "mass": 2.0, "particles": 2}],
    }
    with run_network_simulation(cfg, tmp_path / "out", seed=1, quiet=True) as res:
        df = res.map_positions(1)
        assert (df["reach_index"] == 0).all()
        np.testing.assert_allclose(df["x"], -1000.0)
        np.testing.assert_allclose(df["y"], 500.0)
        res.to_vtp(tmp_path / "vtk", times=[1])
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(tmp_path / "vtk" / "vtp" / "network_0001.vtp"))
    reader.Update()
    assert reader.GetOutput().GetNumberOfPoints() == 2


def test_to_vtp_all_unreleased(tmp_path, run):
    # At output time 0 every particle is unreleased (status 0, s NaN), so write_points finds
    # no valid points and no VTP file is written for that timestep.
    pvd = run.to_vtp(tmp_path / "vtk0", times=[0])
    assert len(list((tmp_path / "vtk0" / "vtp").glob("*.vtp"))) == 0
    assert pvd.exists()


def test_terminal_status_frames(tmp_path):
    from fluvial_particle.network.results import NetworkResults
    from fluvial_particle.network.sources import ParticleSchedule
    from fluvial_particle.network.writer import OUTPUT_FILENAME, NetworkWriter

    # Three particles released in reach 0: one exits (2), one settles (3), one stays active (1).
    sch = ParticleSchedule(
        np.array([0, 0, 0], dtype=np.int32),
        np.array([0.0, 5.0, 10.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0, 0, 0], dtype=np.int32),
    )
    attrs = {"dt": 60.0, "sources": "[]", "mass_units": "kg"}
    path = tmp_path / OUTPUT_FILENAME
    with NetworkWriter(path, n_particles=3, reach_id=np.array([101, 102]), start_time=T0, attrs=attrs) as w:
        w.write_schedule(sch, 0, 3)
        w.write_step(
            0,
            0.0,
            np.array([0, 0, 0], dtype=np.int32),
            np.array([0.0, 5.0, 10.0]),
            np.array([1, 1, 1], dtype=np.int8),
            0,
            3,
        )
        w.write_step(
            1,
            60.0,
            np.array([-1, 0, 0], dtype=np.int32),
            np.array([np.nan, 65.0, 70.0]),
            np.array([2, 3, 1], dtype=np.int8),
            0,
            3,
        )
        w.write_exits(np.array([42.0, 60.0, np.nan]), np.array([1, 0, -1], dtype=np.int32), 0, 3)
    with NetworkResults(path) as res:
        settled = res.terminal(3)
        assert list(settled["particle"]) == [1]
        assert settled.loc[0, "exit_time"] == 60.0 and settled.loc[0, "exit_reach_id"] == 101
        assert settled.loc[0, "mass"] == 2.0 and settled.loc[0, "release_reach_id"] == 101
        arrivals = res.arrival_times()
        assert list(arrivals["particle"]) == [0]
        assert res.terminal(2).equals(arrivals)
        assert res.terminal(4).empty and list(res.terminal(4).columns) == list(arrivals.columns)
        ds = res.positions()
        assert ds["status"].attrs["flag_meanings"] == "unreleased active exited settled removed"
        assert list(ds["status"].attrs["flag_values"]) == [0, 1, 2, 3, 4]
