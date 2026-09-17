"""Tests for the longitudinal dispersion coefficient."""

import numpy as np
import pytest

from fluvial_particle.network.dispersion import dispersion_coefficient, fischer_coefficient


def test_fischer_value():
    k = fischer_coefficient(np.array([1.0]), np.array([2.0]), np.array([10.0]), np.array([0.1]))
    np.testing.assert_allclose(k, 0.011 * 1.0 * 100.0 / (2.0 * 0.1))


def test_fischer_zero_where_dry_or_still():
    v = np.array([1.0, 0.0, 1.0, 1.0])
    d = np.array([1.0, 1.0, 0.0, 1.0])
    w = np.array([10.0, 10.0, 10.0, 10.0])
    u = np.array([0.1, 0.1, 0.1, 0.0])
    k = fischer_coefficient(v, d, w, u)
    assert k[0] > 0
    assert list(k[1:]) == [0.0, 0.0, 0.0]


def test_fischer_scale_and_cap():
    v, d, w, u = (np.array([x]) for x in (1.0, 2.0, 10.0, 0.1))
    base = fischer_coefficient(v, d, w, u)[0]
    assert fischer_coefficient(v, d, w, u, scale=2.0)[0] == pytest.approx(2 * base)
    assert fischer_coefficient(v, d, w, u, cap=1.0)[0] == 1.0


def test_dispersion_models():
    fields = {
        "velocity": np.array([1.0, 1.0]),
        "depth": np.array([2.0, 2.0]),
        "width": np.array([10.0, 10.0]),
        "ustar": np.array([0.1, 0.1]),
        "flow_out": np.array([5.0, 0.0]),
    }
    assert list(dispersion_coefficient(fields, "none")) == [0.0, 0.0]
    k = dispersion_coefficient(fields, "constant", value=3.0)
    assert list(k) == [3.0, 0.0]
    k = dispersion_coefficient(fields, "fischer")
    assert k[0] == pytest.approx(5.5)
    with pytest.raises(ValueError, match="model"):
        dispersion_coefficient(fields, "bogus")
    with pytest.raises(ValueError, match="value"):
        dispersion_coefficient(fields, "constant")


def test_background_is_added_on_wet_reaches_for_every_model():
    fields = {
        "velocity": np.array([1.0, 1.0]),
        "depth": np.array([2.0, 2.0]),
        "width": np.array([10.0, 10.0]),
        "ustar": np.array([0.1, 0.1]),
        "flow_out": np.array([5.0, 0.0]),
    }
    assert list(dispersion_coefficient(fields, "none", background=0.3)) == [0.3, 0.0]
    assert list(dispersion_coefficient(fields, "constant", value=3.0, background=0.3)) == [3.3, 0.0]
    k = dispersion_coefficient(fields, "fischer", background=0.3)
    assert k[0] == pytest.approx(5.8)
    assert k[1] == pytest.approx(5.5)  # Fischer itself is not masked on flow_out; only the background is
    assert list(dispersion_coefficient(fields, "none")) == [0.0, 0.0]  # default adds nothing
