"""Tests for Network topology and polyline mapping."""

import numpy as np
import pytest

from fluvial_particle.network.network import Network, NetworkBins
from tests.network.support import ArrayHydraulicsProvider, chain_dataset, three_reach_dataset


@pytest.fixture
def net():
    return Network(ArrayHydraulicsProvider.from_dataset(three_reach_dataset()).static, crs_wkt="PROJCRS")


def test_basic_attributes(net):
    assert net.n_reach == 3
    assert list(net.reach_id) == [101, 102, 103]
    assert list(net.is_outlet) == [False, False, True]
    assert net.index_of(102) == 1
    assert net.id_of(2) == 103
    assert list(net.index_of_many([103, 101])) == [2, 0]
    with pytest.raises(KeyError):
        net.index_of(999)
    assert net.crs_wkt == "PROJCRS"


def test_topology_queries(net):
    assert list(net.headwaters()) == [101, 102]
    assert list(net.headwaters(as_index=True)) == [0, 1]
    assert list(net.outlets()) == [103]
    assert sorted(net.parents(2)) == [0, 1]
    assert net.parents(0).size == 0
    assert sorted(net.upstream_of(103)) == [101, 102, 103]
    assert list(net.upstream_of(101)) == [101]


def test_map_position_hand_values(net):
    x, y = net.map_position(np.array([2, 2, 2]), np.array([0.0, 1500.0, 3000.0]))
    np.testing.assert_allclose(x, [0.0, 1500.0, 3000.0])
    np.testing.assert_allclose(y, [0.0, 0.0, 0.0])
    # reach 1: polyline length 2 * hypot(1000, 250), hydraulic length 2000; s = 1000 is the middle vertex
    x, y = net.map_position(np.array([1, 1]), np.array([1000.0, 500.0]))
    np.testing.assert_allclose(x, [-1000.0, -1500.0])
    np.testing.assert_allclose(y, [-250.0, -375.0])


def test_map_position_inactive_and_missing_polylines():
    net = Network(ArrayHydraulicsProvider.from_dataset(three_reach_dataset()).static)
    x, y = net.map_position(np.array([-1, 0]), np.array([np.nan, 10.0]))
    assert np.isnan(x[0]) and np.isnan(y[0]) and np.isfinite(x[1])
    bare = Network(ArrayHydraulicsProvider.from_dataset(three_reach_dataset(polylines=None)).static)
    assert not bare.has_polylines
    x, y = bare.map_position(np.array([0]), np.array([10.0]))
    assert np.isnan(x[0]) and np.isnan(y[0])
    assert bare.polylines() == []


def test_polylines_list(net):
    pls = net.polylines()
    assert len(pls) == 3
    np.testing.assert_allclose(pls[2][0], [0.0, 1500.0, 3000.0])


def test_chain_upstream_of():
    net = Network(ArrayHydraulicsProvider.from_dataset(chain_dataset(n_reach=5)).static)
    assert list(net.headwaters()) == [1]
    assert sorted(net.upstream_of(3)) == [1, 2, 3]


def test_bins_layout(net):
    bins = NetworkBins(net, 400.0)
    assert list(bins.bins_per_reach) == [3, 5, 8]
    assert bins.n_bins == 16
    assert list(bins.reach_bin_start) == [0, 3, 8]
    np.testing.assert_allclose(bins.bin_width[:3], 1000.0 / 3)
    np.testing.assert_allclose(bins.bin_width[3:8], 400.0)
    np.testing.assert_allclose(bins.s_start[3:8], [0, 400, 800, 1200, 1600])
    np.testing.assert_allclose(bins.s_end[3:8], [400, 800, 1200, 1600, 2000])
    assert list(bins.bin_reach[8:10]) == [2, 2]


def test_bins_bin_of_edges(net):
    bins = NetworkBins(net, 400.0)
    assert list(bins.bin_of(np.array([1, 1, 2, 0]), np.array([799.9, 800.0, 3000.0, 0.0]))) == [4, 5, 15, 0]


def test_bins_one_per_reach(net):
    bins = NetworkBins(net, np.inf)
    assert bins.n_bins == 3
    assert list(bins.bin_of(np.array([0, 1, 2]), np.array([999.0, 1.0, 2999.0]))) == [0, 1, 2]
    x, _y = bins.midpoints_xy()
    np.testing.assert_allclose(x[2], 1500.0)


def test_topology_validation_at_construction():
    static = dict(ArrayHydraulicsProvider.from_dataset(three_reach_dataset()).static)
    bad_range = {**static, "to_index": np.array([2, 7, -1], dtype=np.int32)}
    with pytest.raises(ValueError, match=r"to_index out of range \[-1, 3\) at reach indices \[1\]"):
        Network(bad_range)
    cyclic = {**static, "to_index": np.array([2, 2, 0], dtype=np.int32)}
    with pytest.raises(ValueError, match="cycle"):
        Network(cyclic)
    bad_length = {**static, "length": np.array([1000.0, 0.0, np.nan])}
    with pytest.raises(ValueError, match=r"length must be positive and finite.*\[1, 2\]"):
        Network(bad_length)


def test_static_mapping_is_defensively_copied():
    static = dict(ArrayHydraulicsProvider.from_dataset(three_reach_dataset()).static)
    net = Network(static)
    static["reach_vertex_count"] = np.array([99, 99, 99], dtype=np.int32)  # mutating after the fact
    np.testing.assert_array_equal(net._static["reach_vertex_count"], [2, 3, 3])


def test_bins_reject_a_non_positive_bin_length(net):
    with pytest.raises(ValueError, match="bin_length must be positive"):
        NetworkBins(net, -1.0)
    with pytest.raises(ValueError, match="bin_length must be positive"):
        NetworkBins(net, 0.0)


def test_map_position_single_vertex_reach_uses_that_vertex():
    ds = three_reach_dataset(
        polylines=[
            [(-1000.0, 500.0)],  # degenerate: one vertex
            [(-2000.0, -500.0), (-1000.0, -250.0), (0.0, 0.0)],
            [(0.0, 0.0), (1500.0, 0.0), (3000.0, 0.0)],
        ]
    )
    net = Network(ArrayHydraulicsProvider.from_dataset(ds).static)
    x, y = net.map_position(np.array([0, 0, 2]), np.array([0.0, 900.0, 1500.0]))
    np.testing.assert_allclose(x, [-1000.0, -1000.0, 1500.0])
    np.testing.assert_allclose(y, [500.0, 500.0, 0.0])


def test_map_position_zero_vertex_reach_falls_back_to_x_mid():
    static = dict(ArrayHydraulicsProvider.from_dataset(three_reach_dataset()).static)
    static["reach_vertex_count"] = np.array([0, 3, 3], dtype=np.int32)  # reach 0 has no polyline
    static["reach_vertex_start"] = np.array([2, 2, 5], dtype=np.int64)
    static["x_mid"] = np.array([-500.0, -1000.0, 1500.0])
    static["y_mid"] = np.array([250.0, -250.0, 0.0])
    net = Network(static)
    x, y = net.map_position(np.array([0, 1, -1]), np.array([100.0, 1000.0, np.nan]))
    np.testing.assert_allclose(x, [-500.0, -1000.0, np.nan])
    np.testing.assert_allclose(y, [250.0, -250.0, np.nan])
    del static["x_mid"], static["y_mid"]  # nothing to fall back on: NaN
    x, _ = Network(static).map_position(np.array([0]), np.array([100.0]))
    assert np.isnan(x[0])
