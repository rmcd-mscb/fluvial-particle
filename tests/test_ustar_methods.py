"""Tests for shear velocity (u*) computation methods."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluvial_particle.RiverGrid import (
    CMU,
    CORE_REQUIRED_FIELDS_2D,
    DEFAULT_WATER_DENSITY,
    GRAVITY,
    USTAR_SOURCE_FIELDS,
    RiverGrid,
)


class TestUstarConstants:
    """Test physical constants used in u* computations."""

    def test_gravity_value(self):
        """Test gravity constant is correct."""
        assert_allclose(GRAVITY, 9.81)

    def test_cmu_value(self):
        """Test C_mu constant for k-epsilon model."""
        assert_allclose(CMU, 0.09)

    def test_default_water_density(self):
        """Test default water density is freshwater value."""
        assert_allclose(DEFAULT_WATER_DENSITY, 1000.0)

    def test_core_required_fields(self):
        """Test that core required fields are defined correctly."""
        assert "bed_elevation" in CORE_REQUIRED_FIELDS_2D
        assert "velocity" in CORE_REQUIRED_FIELDS_2D
        assert "water_surface_elevation" in CORE_REQUIRED_FIELDS_2D

    def test_ustar_source_fields(self):
        """Test that all u* source fields are defined."""
        assert "ustar" in USTAR_SOURCE_FIELDS
        assert "shear_stress" in USTAR_SOURCE_FIELDS
        assert "manning_n" in USTAR_SOURCE_FIELDS
        assert "chezy_c" in USTAR_SOURCE_FIELDS
        assert "darcy_f" in USTAR_SOURCE_FIELDS
        assert "energy_slope" in USTAR_SOURCE_FIELDS
        assert "tke" in USTAR_SOURCE_FIELDS


class TestUstarFormulas:
    """Test u* calculation formulas with known values."""

    def test_ustar_from_shear_stress(self):
        """Test u* = sqrt(tau_b / rho) formula.

        With tau_b = 10 Pa and rho = 1000 kg/m³:
        u* = sqrt(10 / 1000) = sqrt(0.01) = 0.1 m/s
        """
        tau_b = 10.0  # Pa
        rho = 1000.0  # kg/m³
        expected_ustar = np.sqrt(tau_b / rho)
        assert_allclose(expected_ustar, 0.1)

    def test_ustar_from_manning(self):
        """Test u* = U * n * sqrt(g) / h^(1/6) formula.

        With U = 1.0 m/s, n = 0.03, h = 1.0 m:
        u* = 1.0 * 0.03 * sqrt(9.81) / 1.0^(1/6)
        u* = 0.03 * 3.132 / 1.0 = 0.094 m/s
        """
        vel_mag = 1.0  # m/s
        n = 0.03  # Manning's n
        h = 1.0  # m
        expected_ustar = vel_mag * n * np.sqrt(GRAVITY) / np.power(h, 1.0 / 6.0)
        assert_allclose(expected_ustar, 0.03 * np.sqrt(9.81), rtol=1e-10)

    def test_ustar_from_chezy(self):
        """Test u* = U * sqrt(g) / C formula.

        With U = 1.0 m/s, C = 50 m^0.5/s:
        u* = 1.0 * sqrt(9.81) / 50 = 3.132 / 50 = 0.0627 m/s
        """
        vel_mag = 1.0  # m/s
        chezy_c = 50.0  # Chezy C
        expected_ustar = vel_mag * np.sqrt(GRAVITY) / chezy_c
        assert_allclose(expected_ustar, np.sqrt(9.81) / 50.0, rtol=1e-10)

    def test_ustar_from_darcy_weisbach(self):
        """Test u* = U * sqrt(f / 8) formula.

        With U = 1.0 m/s, f = 0.02:
        u* = 1.0 * sqrt(0.02 / 8) = sqrt(0.0025) = 0.05 m/s
        """
        vel_mag = 1.0  # m/s
        darcy_f = 0.02  # Darcy-Weisbach friction factor
        expected_ustar = vel_mag * np.sqrt(darcy_f / 8.0)
        assert_allclose(expected_ustar, 0.05)

    def test_ustar_from_energy_slope(self):
        """Test u* = sqrt(g * h * S) formula.

        With h = 1.0 m, S = 0.001:
        u* = sqrt(9.81 * 1.0 * 0.001) = sqrt(0.00981) = 0.099 m/s
        """
        h = 1.0  # m
        slope = 0.001  # energy slope
        expected_ustar = np.sqrt(GRAVITY * h * slope)
        assert_allclose(expected_ustar, np.sqrt(0.00981), rtol=1e-10)

    def test_ustar_from_tke(self):
        """Test u* = C_mu^(1/4) * sqrt(k) formula.

        With k = 0.1 m²/s² and C_mu = 0.09:
        u* = 0.09^0.25 * sqrt(0.1) = 0.5477 * 0.3162 = 0.173 m/s
        """
        k = 0.1  # m²/s²
        expected_ustar = np.power(CMU, 0.25) * np.sqrt(k)
        # CMU^0.25 = 0.09^0.25 ≈ 0.5477
        assert_allclose(expected_ustar, np.power(0.09, 0.25) * np.sqrt(0.1), rtol=1e-10)


class TestUstarMethodSelection:
    """Test u* method selection and validation in RiverGrid."""

    def test_shear_stress_method_selected(self):
        """Test that shear_stress method is selected when provided."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "shear_stress": "ShearStress (magnitude)",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
        )
        assert river._ustar_method == "shear_stress"

    def test_manning_scalar_method_selected(self):
        """Test that manning method is selected when scalar provided."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            manning_n=0.03,
        )
        assert river._ustar_method == "manning"
        assert river._manning_n_scalar == 0.03

    def test_chezy_scalar_method_selected(self):
        """Test that chezy method is selected when scalar provided."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            chezy_c=50.0,
        )
        assert river._ustar_method == "chezy"
        assert river._chezy_c_scalar == 50.0

    def test_darcy_scalar_method_selected(self):
        """Test that darcy method is selected when scalar provided."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            darcy_f=0.02,
        )
        assert river._ustar_method == "darcy"
        assert river._darcy_f_scalar == 0.02

    def test_explicit_ustar_method_override(self):
        """Test that ustar_method setting overrides auto-selection."""
        # Provide both shear_stress and manning_n, but force manning
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "shear_stress": "ShearStress (magnitude)",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            manning_n=0.03,
            ustar_method="manning",
        )
        assert river._ustar_method == "manning"

    def test_no_ustar_source_raises_error(self):
        """Test that missing u* source raises ValueError."""
        with pytest.raises(ValueError, match="No method to compute shear velocity"):
            RiverGrid(
                track3d=0,
                filename2d="./tests/data/Result_straight_2d_1.vtk",
                field_map_2d={
                    "bed_elevation": "Elevation",
                    "velocity": "Velocity",
                    "water_surface_elevation": "WaterSurfaceElevation",
                    # No u* source provided!
                },
                field_map_3d={"velocity": "Velocity"},
            )

    def test_invalid_ustar_method_raises_error(self):
        """Test that invalid ustar_method raises ValueError."""
        with pytest.raises(ValueError, match="ustar_method='invalid' requested but not available"):
            RiverGrid(
                track3d=0,
                filename2d="./tests/data/Result_straight_2d_1.vtk",
                field_map_2d={
                    "bed_elevation": "Elevation",
                    "shear_stress": "ShearStress (magnitude)",
                    "velocity": "Velocity",
                    "water_surface_elevation": "WaterSurfaceElevation",
                },
                field_map_3d={"velocity": "Velocity"},
                ustar_method="invalid",
            )

    def test_custom_water_density(self):
        """Test that custom water density is stored correctly."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "shear_stress": "ShearStress (magnitude)",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            water_density=1025.0,  # seawater
        )
        assert river._water_density == 1025.0


class TestUstarPriorityOrder:
    """Test the priority order of u* method selection."""

    def test_shear_stress_over_manning(self):
        """Test that shear_stress takes priority over manning_n."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "shear_stress": "ShearStress (magnitude)",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            manning_n=0.03,  # Also provided, but should be lower priority
        )
        assert river._ustar_method == "shear_stress"

    def test_manning_over_chezy(self):
        """Test that manning_n takes priority over chezy_c."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            manning_n=0.03,
            chezy_c=50.0,  # Also provided, but lower priority
        )
        assert river._ustar_method == "manning"

    def test_chezy_over_darcy(self):
        """Test that chezy_c takes priority over darcy_f."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            chezy_c=50.0,
            darcy_f=0.02,  # Also provided, but lower priority
        )
        assert river._ustar_method == "chezy"


class TestRequiredKeysProperty:
    """Test the required_keys2d property returns correct fields."""

    def test_required_keys_with_shear_stress(self):
        """Test required_keys2d includes shear_stress when method is shear_stress."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "shear_stress": "ShearStress (magnitude)",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
        )
        required = river.required_keys2d
        assert "bed_elevation" in required
        assert "velocity" in required
        assert "water_surface_elevation" in required
        assert "shear_stress" in required

    def test_required_keys_with_manning_scalar(self):
        """Test required_keys2d excludes field when using scalar manning."""
        river = RiverGrid(
            track3d=0,
            filename2d="./tests/data/Result_straight_2d_1.vtk",
            field_map_2d={
                "bed_elevation": "Elevation",
                "velocity": "Velocity",
                "water_surface_elevation": "WaterSurfaceElevation",
            },
            field_map_3d={"velocity": "Velocity"},
            manning_n=0.03,
        )
        required = river.required_keys2d
        assert "bed_elevation" in required
        assert "velocity" in required
        assert "water_surface_elevation" in required
        # manning_n is scalar, so not in required fields
        assert "manning_n" not in required
        assert "shear_stress" not in required
