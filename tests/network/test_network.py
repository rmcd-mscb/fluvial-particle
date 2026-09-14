"""Tests for Network topology and polyline mapping."""

import numpy as np
import pytest

from fluvial_particle.network.network import Network
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
