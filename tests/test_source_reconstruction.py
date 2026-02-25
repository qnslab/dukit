# -*- coding: utf-8 -*-
"""Smoke tests for source reconstruction functionality.

These tests verify basic functionality and API compliance. They don't
validate physical correctness, just that functions run without errors
and return data in expected formats.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal


class TestGetMagnetizationFromBxyz:
    """Smoke tests for get_magnetization_from_bxyz function."""

    def test_basic_call_xy_components(self):
        """Test that function runs with Bx, By inputs."""
        from dukit.source import get_magnetization_from_bxyz
        
        shape = (32, 32)
        Bx = np.random.randn(*shape) * 10.0
        By = np.random.randn(*shape) * 5.0
        
        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=1.5e-6,
            standoff=100e-9,
            use_components="xy",
        )
        
        assert "Mz" in result
        assert "_metadata" in result
        assert result["Mz"].shape == shape
        assert np.all(np.isfinite(result["Mz"]))

    def test_basic_call_z_component(self):
        """Test that function runs with Bz input."""
        from dukit.source import get_magnetization_from_bxyz
        
        shape = (32, 32)
        Bz = np.random.randn(*shape) * 10.0
        
        result = get_magnetization_from_bxyz(
            Bx=None, By=None, Bz=Bz,
            pixel_size=1.5e-6,
            standoff=100e-9,
            use_components="z",
        )
        
        assert "Mz" in result
        assert "_metadata" in result
        assert result["Mz"].shape == shape
        assert np.all(np.isfinite(result["Mz"]))

    def test_auto_mode_selects_xy(self):
        """Test that auto mode selects xy when Bx, By provided."""
        from dukit.source import get_magnetization_from_bxyz
        
        shape = (32, 32)
        Bx = np.random.randn(*shape) * 10.0
        By = np.random.randn(*shape) * 5.0
        
        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=1.5e-6,
            use_components="auto",
        )
        
        assert "Mz" in result
        assert result["_metadata"]["use_components"] == "xy"

    def test_auto_mode_requires_input(self):
        """Test that auto mode raises error when no field provided."""
        from dukit.source import get_magnetization_from_bxyz
        
        with pytest.raises(ValueError, match="Must provide"):
            get_magnetization_from_bxyz(
                Bx=None, By=None, Bz=None,
                pixel_size=1.5e-6,
                use_components="auto",
            )

    def test_requires_bz_for_z_mode(self):
        """Test that z mode raises error when Bz not provided."""
        from dukit.source import get_magnetization_from_bxyz
        
        Bx = np.random.randn(16, 16)
        By = np.random.randn(16, 16)
        
        with pytest.raises(ValueError, match="Bz must be provided"):
            get_magnetization_from_bxyz(
                Bx, By, Bz=None,
                pixel_size=1.5e-6,
                use_components="z",
            )

    def test_invalid_use_components(self):
        """Test that invalid use_components raises error."""
        from dukit.source import get_magnetization_from_bxyz
        
        Bx = np.random.randn(16, 16)
        By = np.random.randn(16, 16)
        
        with pytest.raises(ValueError, match="Invalid use_components"):
            get_magnetization_from_bxyz(
                Bx, By,
                pixel_size=1.5e-6,
                use_components="invalid",
            )

    def test_optional_parameters(self):
        """Test that optional parameters are accepted and stored in metadata."""
        from dukit.source import get_magnetization_from_bxyz
        
        shape = (16, 16)
        Bx = np.random.randn(*shape)
        By = np.random.randn(*shape)
        
        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=2.0e-6,
            standoff=150e-9,
            nv_layer_thickness=10e-9,
            pad_mode="reflect",
            pad_factor=3,
            k_vector_epsilon=1e-5,
            do_hanning_filter=True,
            hanning_low_cutoff=100e-9,
            hanning_high_cutoff=500e-9,
        )
        
        meta = result["_metadata"]
        assert meta["pixel_size"] == 2.0e-6
        assert meta["standoff"] == 150e-9
        assert meta["nv_layer_thickness"] == 10e-9
        assert meta["pad_mode"] == "reflect"
        assert meta["pad_factor"] == 3
        assert meta["do_hanning_filter"] is True
        assert meta["hanning_low_cutoff"] == 100e-9
        assert meta["hanning_high_cutoff"] == 500e-9

    def test_output_units(self):
        """Test that output magnetization is finite and reasonable."""
        from dukit.source import get_magnetization_from_bxyz
        
        # Create varying field (gradient, not uniform, to get non-zero response)
        shape = (16, 16)
        x = np.linspace(-1, 1, shape[1])
        X, _ = np.meshgrid(x, np.linspace(-1, 1, shape[0]))
        Bx = X * 10.0  # gradient in x direction
        By = np.zeros(shape)
        
        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=1.5e-6,
            standoff=100e-9,
        )
        
        # Just check we get finite values in reasonable range
        Mz = result["Mz"]
        assert np.all(np.isfinite(Mz))
        assert np.max(np.abs(Mz)) < 1e6  # Should not be orders of magnitude too high

    def test_zero_field_output(self):
        """Test that zero input field produces near-zero output."""
        from dukit.source import get_magnetization_from_bxyz
        
        shape = (16, 16)
        Bx = np.zeros(shape)
        By = np.zeros(shape)
        
        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=1.5e-6,
            standoff=100e-9,
        )
        
        # Zero field should give near-zero magnetization
        assert_allclose(result["Mz"], 0.0, atol=0.1)

    def test_uniform_field_output(self):
        """Test that uniform input field produces near-zero output (DC component lost)."""
        from dukit.source import get_magnetization_from_bxyz
        
        shape = (16, 16)
        # 10 G uniform field
        Bx = np.ones(shape) * 10.0
        By = np.ones(shape) * 5.0
        
        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=1.5e-6,
            standoff=100e-9,
        )
        
        # Uniform field should give near-zero magnetization (DC component lost in FT)
        assert_allclose(result["Mz"], 0.0, atol=0.1)


class TestGetMagnetizationFromBdefect:
    """Smoke tests for get_magnetization_from_bdefect function."""

    def test_basic_call(self):
        """Test that function runs with basic inputs."""
        from dukit.source import get_magnetization_from_bdefect
        
        shape = (32, 32)
        b_defect = np.random.randn(*shape) * 10.0
        u_defect = np.array([0.0, 0.0, 1.0])  # z-direction
        
        result = get_magnetization_from_bdefect(
            b_defect,
            u_defect,
            pixel_size=1.5e-6,
            standoff=100e-9,
        )
        
        assert "Mz" in result
        assert "_metadata" in result
        assert result["Mz"].shape == shape
        assert np.all(np.isfinite(result["Mz"]))

    def test_u_defect_shape_validation(self):
        """Test that u_defect must be 3-component vector."""
        from dukit.source import get_magnetization_from_bdefect
        
        b_defect = np.ones((16, 16))
        u_defect = np.array([1.0, 0.0])  # Wrong shape!
        
        # This should not raise a shape error immediately
        # but might have numerical issues later
        try:
            result = get_magnetization_from_bdefect(
                b_defect,
                u_defect,
                pixel_size=1.5e-6,
            )
            # If it runs, just check output format
            assert "Mz" in result
        except (IndexError, ValueError):
            # Also acceptable if shape validation catches it
            pass

    def test_nv_above_sample_option(self):
        """Test that nv_above_sample option is accepted."""
        from dukit.source import get_magnetization_from_bdefect
        
        shape = (16, 16)
        b_defect = np.random.randn(*shape) * 10.0
        u_defect = np.array([0.0, 0.0, 1.0])
        
        # True (default)
        result1 = get_magnetization_from_bdefect(
            b_defect, u_defect,
            pixel_size=1.5e-6,
            nv_above_sample=True,
        )
        assert "Mz" in result1
        
        # False
        result2 = get_magnetization_from_bdefect(
            b_defect, u_defect,
            pixel_size=1.5e-6,
            nv_above_sample=False,
        )
        assert "Mz" in result2

    def test_metadata_contains_options(self):
        """Test that metadata includes nv_above_sample option."""
        from dukit.source import get_magnetization_from_bdefect
        
        shape = (16, 16)
        b_defect = np.ones(shape)
        u_defect = np.array([0.0, 0.0, 1.0])
        
        result = get_magnetization_from_bdefect(
            b_defect, u_defect,
            pixel_size=2.0e-6,
            standoff=150e-9,
            nv_above_sample=False,
        )
        
        meta = result["_metadata"]
        assert meta["nv_above_sample"] is False
        assert meta["standoff"] == 150e-9

    def test_zero_defect_output(self):
        """Test that zero defect field produces near-zero output."""
        from dukit.source import get_magnetization_from_bdefect
        
        shape = (16, 16)
        b_defect = np.zeros(shape)
        u_defect = np.array([0.0, 0.0, 1.0])
        
        result = get_magnetization_from_bdefect(
            b_defect, u_defect,
            pixel_size=1.5e-6,
            standoff=100e-9,
        )
        
        assert_allclose(result["Mz"], 0.0, atol=0.1)


class TestReconstructSource:
    """Smoke tests for reconstruct_source high-level interface."""

    def test_reconstruct_magnetization_from_bxyz(self):
        """Test high-level magnetization reconstruction from Bxyz."""
        from dukit.source import reconstruct_source
        
        shape = (32, 32)
        bxyz = {
            "Bx": np.random.randn(*shape) * 10.0,
            "By": np.random.randn(*shape) * 5.0,
            "Bz": np.random.randn(*shape) * 3.0,
        }
        
        result = reconstruct_source(
            bxyz,
            pixel_size=1.5e-6,
            source_type="magnetization",
            standoff=100e-9,
        )
        
        assert "Mz" in result
        assert "_metadata" in result
        assert result["Mz"].shape == shape
        assert np.all(np.isfinite(result["Mz"]))

    def test_reconstruct_magnetization_requires_bxy(self):
        """Test that magnetization reconstruction requires Bx, By."""
        from dukit.source import reconstruct_source
        
        # Missing Bx
        bxyz = {"By": np.random.randn(16, 16)}
        
        with pytest.raises(ValueError, match="must contain"):
            reconstruct_source(
                bxyz,
                pixel_size=1.5e-6,
                source_type="magnetization",
            )

    def test_invalid_source_type(self):
        """Test that invalid source_type raises error."""
        from dukit.source import reconstruct_source
        
        bxyz = {
            "Bx": np.random.randn(16, 16),
            "By": np.random.randn(16, 16),
        }
        
        with pytest.raises(ValueError, match="Invalid source_type"):
            reconstruct_source(
                bxyz,
                pixel_size=1.5e-6,
                source_type="invalid_type",
            )

    def test_fourier_kwargs_passed_through(self):
        """Test that Fourier kwargs are passed to low-level function."""
        from dukit.source import reconstruct_source
        
        shape = (16, 16)
        bxyz = {
            "Bx": np.random.randn(*shape),
            "By": np.random.randn(*shape),
        }
        
        result = reconstruct_source(
            bxyz,
            pixel_size=1.5e-6,
            source_type="magnetization",
            standoff=100e-9,
            pad_mode="reflect",
            pad_factor=3,
            k_vector_epsilon=1e-5,
            do_hanning_filter=True,
        )
        
        meta = result["_metadata"]
        assert meta["pad_mode"] == "reflect"
        assert meta["pad_factor"] == 3
        assert meta["do_hanning_filter"] is True

    def test_metadata_present(self):
        """Test that metadata is returned by high-level interface."""
        from dukit.source import reconstruct_source
        
        shape = (16, 16)
        bxyz = {
            "Bx": np.random.randn(*shape),
            "By": np.random.randn(*shape),
        }
        
        result = reconstruct_source(
            bxyz,
            pixel_size=1.5e-6,
            source_type="magnetization",
        )
        
        assert "_metadata" in result
        assert result["_metadata"]["pixel_size"] == 1.5e-6


class TestIntegration:
    """Integration tests for source reconstruction workflows."""

    def test_full_magnetization_workflow(self):
        """Test magnetization reconstruction from Bxyz input."""
        from dukit.source import reconstruct_source
        
        # Simulate B-field from field reconstruction
        shape = (32, 32)
        bxyz = {
            "Bx": np.random.randn(*shape) * 10.0,
            "By": np.random.randn(*shape) * 5.0,
            "Bz": np.random.randn(*shape) * 3.0,
        }
        
        # Reconstruct magnetization
        result = reconstruct_source(
            bxyz,
            pixel_size=1.5e-6,
            source_type="magnetization",
            standoff=100e-9,
            use_components="xy",
        )
        
        # Check output
        assert "Mz" in result
        assert "_metadata" in result
        assert np.all(np.isfinite(result["Mz"]))
        assert result["_metadata"]["use_components"] == "xy"

    def test_magnetization_from_defect_workflow(self):
        """Test magnetization reconstruction from single defect."""
        from dukit.source import get_magnetization_from_bdefect
        
        # Simulate B_defect measurement
        shape = (32, 32)
        b_defect = np.random.randn(*shape) * 10.0
        u_defect = np.array([0.577, 0.577, 0.577])  # <111> direction
        
        result = get_magnetization_from_bdefect(
            b_defect,
            u_defect,
            pixel_size=1.5e-6,
            standoff=100e-9,
        )
        
        # Check output format
        assert "Mz" in result
        assert "_metadata" in result
        assert result["Mz"].shape == shape
        assert np.all(np.isfinite(result["Mz"]))

    def test_both_interfaces_return_same_type(self):
        """Test that high and low-level interfaces return same dict format."""
        from dukit.source import reconstruct_source, get_magnetization_from_bxyz
        
        shape = (16, 16)
        Bx = np.random.randn(*shape)
        By = np.random.randn(*shape)
        
        bxyz = {"Bx": Bx, "By": By}
        
        result1 = reconstruct_source(
            bxyz,
            pixel_size=1.5e-6,
            source_type="magnetization",
        )
        
        result2 = get_magnetization_from_bxyz(Bx, By, pixel_size=1.5e-6)
        
        # Both should have Mz and _metadata
        assert "Mz" in result1
        assert "Mz" in result2
        assert "_metadata" in result1
        assert "_metadata" in result2
        assert result1["Mz"].shape == result2["Mz"].shape

    def test_metadata_tracking(self):
        """Test that reconstruction parameters are tracked in metadata."""
        from dukit.source import get_magnetization_from_bxyz
        
        shape = (16, 16)
        Bx = np.random.randn(*shape)
        By = np.random.randn(*shape)
        
        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=2.5e-6,
            standoff=200e-9,
            nv_layer_thickness=15e-9,
            use_components="xy",
            pad_mode="edge",
            pad_factor=2,
            do_hanning_filter=False,
        )
        
        meta = result["_metadata"]
        
        assert meta["pixel_size"] == 2.5e-6
        assert meta["standoff"] == 200e-9
        assert meta["nv_layer_thickness"] == 15e-9
        assert meta["use_components"] == "xy"
        assert meta["pad_mode"] == "edge"
        assert meta["pad_factor"] == 2
        assert meta["do_hanning_filter"] is False

    def test_consistency_between_functions(self):
        """Test that reconstruction functions have consistent API."""
        from dukit.source import (
            get_magnetization_from_bxyz,
            get_magnetization_from_bdefect,
        )
        
        shape = (16, 16)
        
        # Both functions should accept similar optional parameters
        Bx, By = np.random.randn(*shape), np.random.randn(*shape)
        b_defect = np.random.randn(*shape)
        u_defect = np.array([0.0, 0.0, 1.0])
        
        result1 = get_magnetization_from_bxyz(Bx, By, pixel_size=1.5e-6)
        result2 = get_magnetization_from_bdefect(b_defect, u_defect, pixel_size=1.5e-6)
        
        # Both return dict with Mz and _metadata
        for result in [result1, result2]:
            assert "Mz" in result
            assert "_metadata" in result
            assert isinstance(result["Mz"], np.ndarray)
            assert isinstance(result["_metadata"], dict)


class TestInPlaneMagnetization:
    """Tests for in-plane magnetization reconstruction."""

    def test_in_plane_magnetization_from_bxyz(self):
        """Test that in-plane magnetization (Mpsi) can be reconstructed."""
        from dukit.source import get_magnetization_from_bxyz

        shape = (32, 32)
        Bx = np.random.randn(*shape) * 10.0
        By = np.random.randn(*shape) * 5.0

        # In-plane at 45 degrees
        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=1.5e-6,
            standoff=100e-9,
            magnetization_angle=45.0,
        )

        # Should return Mpsi, not Mz
        assert "Mpsi" in result
        assert "Mz" not in result
        assert "_metadata" in result
        assert result["Mpsi"].shape == shape
        assert np.all(np.isfinite(result["Mpsi"]))

    def test_out_of_plane_magnetization_returns_mz(self):
        """Test that out-of-plane magnetization (magnetization_angle=None) returns Mz."""
        from dukit.source import get_magnetization_from_bxyz

        shape = (32, 32)
        Bx = np.random.randn(*shape) * 10.0
        By = np.random.randn(*shape) * 5.0

        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=1.5e-6,
            magnetization_angle=None,  # Out-of-plane
        )

        # Should return Mz, not Mpsi
        assert "Mz" in result
        assert "Mpsi" not in result

    def test_in_plane_from_bdefect(self):
        """Test in-plane magnetization from single defect measurement."""
        from dukit.source import get_magnetization_from_bdefect

        shape = (32, 32)
        b_defect = np.random.randn(*shape) * 10.0
        u_defect = np.array([0.707, 0.707, 0.0])  # In-plane direction

        result = get_magnetization_from_bdefect(
            b_defect,
            u_defect,
            pixel_size=1.5e-6,
            standoff=100e-9,
            magnetization_angle=30.0,
        )

        # Should return Mpsi for in-plane angle
        assert "Mpsi" in result
        assert "Mz" not in result
        assert result["Mpsi"].shape == shape
        assert np.all(np.isfinite(result["Mpsi"]))

    def test_in_plane_magnetization_metadata(self):
        """Test that metadata includes magnetization_angle."""
        from dukit.source import get_magnetization_from_bxyz

        shape = (16, 16)
        Bx = np.random.randn(*shape)
        By = np.random.randn(*shape)

        result = get_magnetization_from_bxyz(
            Bx, By,
            pixel_size=1.5e-6,
            magnetization_angle=45.0,
        )

        meta = result["_metadata"]
        assert meta["magnetization_angle"] == 45.0

    def test_in_plane_different_angles(self):
        """Test that different in-plane angles work."""
        from dukit.source import get_magnetization_from_bxyz

        shape = (16, 16)
        Bx = np.random.randn(*shape) * 10.0
        By = np.random.randn(*shape) * 5.0

        for angle in [0.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]:
            result = get_magnetization_from_bxyz(
                Bx, By,
                pixel_size=1.5e-6,
                magnetization_angle=angle,
            )

            assert "Mpsi" in result
            assert np.all(np.isfinite(result["Mpsi"]))

    def test_in_plane_z_component(self):
        """Test in-plane magnetization using Bz component."""
        from dukit.source import get_magnetization_from_bxyz

        shape = (32, 32)
        Bz = np.random.randn(*shape) * 10.0

        result = get_magnetization_from_bxyz(
            Bx=None, By=None, Bz=Bz,
            pixel_size=1.5e-6,
            standoff=100e-9,
            use_components="z",
            magnetization_angle=45.0,
        )

        # Should return Mpsi for in-plane angle
        assert "Mpsi" in result
        assert np.all(np.isfinite(result["Mpsi"]))


class TestNormalizeInPlaneMag:
    """Tests for normalize_in_plane_mag function."""

    def test_normalize_in_plane_mag_exists(self):
        """Test that normalize_in_plane_mag can be imported and called."""
        from dukit.source import normalize_in_plane_mag

        mag = np.random.randn(32, 32)
        result = normalize_in_plane_mag(mag, angle_deg=30.0)

        assert result.shape == mag.shape
        assert np.all(np.isfinite(result))

    def test_normalize_preserves_shape(self):
        """Test that normalization preserves input shape."""
        from dukit.source import normalize_in_plane_mag

        for shape in [(16, 16), (32, 64), (100, 100)]:
            mag = np.random.randn(*shape)
            result = normalize_in_plane_mag(mag, angle_deg=45.0)

            assert result.shape == shape

    def test_normalize_various_angles(self):
        """Test normalization at various angles."""
        from dukit.source import normalize_in_plane_mag

        mag = np.random.randn(32, 32)

        for angle in [0.0, 30.0, 45.0, 60.0, 90.0, 135.0, -45.0]:
            result = normalize_in_plane_mag(mag, angle_deg=angle)

            assert result.shape == mag.shape
            assert np.all(np.isfinite(result))

    def test_normalize_with_edge_pixels(self):
        """Test edge_pixels parameter."""
        from dukit.source import normalize_in_plane_mag

        mag = np.random.randn(32, 32)

        result1 = normalize_in_plane_mag(mag, angle_deg=45.0, edge_pixels=5)
        result2 = normalize_in_plane_mag(mag, angle_deg=45.0, edge_pixels=20)

        # Both should work and have correct shape
        assert result1.shape == mag.shape
        assert result2.shape == mag.shape
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))

    def test_normalize_does_not_modify_input(self):
        """Test that input array is not modified."""
        from dukit.source import normalize_in_plane_mag

        mag = np.random.randn(32, 32)
        mag_copy = mag.copy()

        _ = normalize_in_plane_mag(mag, angle_deg=30.0)

        assert_array_almost_equal(mag, mag_copy)

    def test_normalize_returns_finite(self):
        """Test that normalization always returns finite values."""
        from dukit.source import normalize_in_plane_mag

        for _ in range(10):
            mag = np.random.randn(50, 50) * 100.0
            result = normalize_in_plane_mag(mag, angle_deg=np.random.uniform(-180, 180))

            assert np.all(np.isfinite(result))

    def test_normalize_with_negative_angle(self):
        """Test normalization with negative angles."""
        from dukit.source import normalize_in_plane_mag

        mag = np.random.randn(32, 32)

        result = normalize_in_plane_mag(mag, angle_deg=-45.0)

        assert result.shape == mag.shape
        assert np.all(np.isfinite(result))


    def test_default_magnetization_angle_is_out_of_plane(self):
        """Test that default magnetization_angle=None gives Mz (out-of-plane)."""
        from dukit.source import get_magnetization_from_bxyz

        shape = (16, 16)
        Bx = np.random.randn(*shape)
        By = np.random.randn(*shape)

        # Not specifying magnetization_angle (defaults to None)
        result = get_magnetization_from_bxyz(Bx, By, pixel_size=1.5e-6)

        # Should return Mz (out-of-plane)
        assert "Mz" in result
        assert "Mpsi" not in result
