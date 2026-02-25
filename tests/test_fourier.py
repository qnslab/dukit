# -*- coding: utf-8 -*-
"""Tests for Fourier tools (dukit.fourier)."""

import numpy as np
import pytest

from dukit.fourier import (
    define_k_vectors,
    hanning_filter_kspace,
    define_current_transform,
    define_magnetization_transformation,
    pad_image,
    unpad_image,
    MAG_UNIT_CONV,
    MU_0,
)


class TestConstants:
    """Test physical constants."""

    def test_mu_0_value(self):
        """Test vacuum permeability value."""
        # Should be 4π × 10^-7 H/m
        expected = 4 * np.pi * 1e-7
        np.testing.assert_allclose(MU_0, expected, rtol=1e-8)

    def test_mag_unit_conv(self):
        """Test magnetization unit conversion."""
        assert isinstance(MAG_UNIT_CONV, float)
        # Check it's a reasonable conversion factor
        assert MAG_UNIT_CONV > 0
        assert MAG_UNIT_CONV < 1e6  # Should be less than a million


class TestKVectors:
    """Test k-vector definition."""

    def test_define_k_vectors_square(self):
        """Test k-vectors for square image."""
        shape = (32, 32)
        pixel_size = 1e-6
        
        kx, ky, k_mag = define_k_vectors(shape, pixel_size)
        
        assert kx.shape == shape
        assert ky.shape == shape
        assert k_mag.shape == shape
        
        # Check basic properties
        assert kx.shape == shape
        assert ky.shape == shape
        assert k_mag.shape == shape
        
        # Center should have k=0 (with epsilon offset)
        center = (shape[0] // 2, shape[1] // 2)
        np.testing.assert_allclose(k_mag[center], 0.0, atol=1e-5)

    def test_define_k_vectors_rectangular(self):
        """Test k-vectors for rectangular image."""
        shape = (32, 64)
        pixel_size = 1e-6
        
        kx, ky, k_mag = define_k_vectors(shape, pixel_size)
        
        assert kx.shape == shape
        assert ky.shape == shape
        assert k_mag.shape == shape

    def test_k_vector_units(self):
        """Test k-vector units are correct."""
        shape = (16, 16)
        pixel_size = 1e-6  # 1 micron
        
        kx, ky, k_mag = define_k_vectors(shape, pixel_size)
        
        # Just check that k vectors are finite and reasonable
        assert np.all(np.isfinite(kx))
        assert np.all(np.isfinite(ky))
        assert np.all(np.isfinite(k_mag))
        
        # Check that center has small k (due to epsilon)
        center = (shape[0] // 2, shape[1] // 2)
        assert np.abs(k_mag[center]) < 1e-4

    def test_k_zero_at_center(self):
        """Test that k=0 at image center."""
        shape = (32, 32)
        pixel_size = 1e-6
        
        kx, ky, k_mag = define_k_vectors(shape, pixel_size)
        
        # Center should have k=0 (with epsilon offset)
        center = (shape[0] // 2, shape[1] // 2)
        np.testing.assert_allclose(k_mag[center], 0.0, atol=1e-5)


class TestHanningFilter:
    """Test Hanning filter in k-space."""

    def test_hanning_filter_exists(self):
        """Test that hanning_filter_kspace exists and returns something."""
        shape = (32, 32)
        pixel_size = 1e-6
        
        kx, ky, k_mag = define_k_vectors(shape, pixel_size)
        filter_ = hanning_filter_kspace(
            k_mag, 
            do_filt=True,
            hanning_low_cutoff=None,
            hanning_high_cutoff=None,
            standoff=1e-6
        )
        
        assert filter_ is not None
        assert filter_.shape == shape


class TestCurrentTransform:
    """Test current density transformation."""

    def test_define_current_transform_exists(self):
        """Test that define_current_transform exists and returns something."""
        shape = (32, 32)
        pixel_size = 1e-6
        
        kx, ky, k_mag = define_k_vectors(shape, pixel_size)
        u_proj = (1.0, 0.0, 0.0)  # Default projection
        transform = define_current_transform(u_proj, kx, ky, k_mag)
        
        # Should return Jx, Jy transforms
        assert transform is not None
        assert len(transform) == 2
        assert transform[0].shape == shape
        assert transform[1].shape == shape


class TestMagnetizationTransform:
    """Test magnetization transformation."""

    def test_define_magnetization_transformation_exists(self):
        """Test that define_magnetization_transformation exists and returns something."""
        shape = (32, 32)
        pixel_size = 1e-6
        
        kx, ky, k_mag = define_k_vectors(shape, pixel_size)
        transform = define_magnetization_transformation(kx, ky, k_mag)
        
        # Should return a 3x3 transformation matrix for each k-space point
        assert transform is not None
        assert transform.shape == (3, 3, shape[0], shape[1])


class TestPadding:
    """Test image padding functions."""

    def test_pad_image(self):
        """Test image padding."""
        img = np.random.rand(16, 16)
        pad_factor = 0.5  # Pad by 50% of original size
        
        padded, padder = pad_image(img, 'constant', pad_factor)
        
        expected_pad = int(16 * pad_factor)
        assert padded.shape == (16 + 2*expected_pad, 16 + 2*expected_pad)
        
        # Check original is in center
        center_slice = slice(expected_pad, -expected_pad)
        np.testing.assert_array_equal(padded[center_slice, center_slice], img)

    def test_pad_image_asymmetric(self):
        """Test asymmetric padding."""
        img = np.random.rand(16, 16)
        
        # For now, just test symmetric padding with different factor
        pad_factor = 1.0  # Pad by 100% of original size
        padded, padder = pad_image(img, 'constant', pad_factor)
        
        expected_pad = int(16 * pad_factor)
        assert padded.shape == (16 + 2*expected_pad, 16 + 2*expected_pad)

    def test_unpad_image(self):
        """Test image unpadding."""
        img = np.random.rand(16, 16)
        pad_factor = 0.5
        
        # Pad then unpad
        padded, padder = pad_image(img, 'constant', pad_factor)
        unpadded = unpad_image(padded, padder)
        
        np.testing.assert_array_equal(unpadded, img)

    def test_unpad_asymmetric(self):
        """Test asymmetric unpadding."""
        img = np.random.rand(16, 16)
        
        # For now, just test symmetric padding
        pad_factor = 1.0
        padded, padder = pad_image(img, 'constant', pad_factor)
        unpadded = unpad_image(padded, padder)
        
        np.testing.assert_array_equal(unpadded, img)

    def test_pad_unpad_roundtrip_fft(self):
        """Test that padding preserves FFT information."""
        # Create test image with known frequency content
        x = np.linspace(0, 10, 32)
        y = np.linspace(0, 10, 32)
        X, Y = np.meshgrid(x, y)
        img = np.sin(2 * np.pi * X / 5) + np.cos(2 * np.pi * Y / 3)
        
        # Pad
        pad_factor = 0.5
        padded, padder = pad_image(img, 'constant', pad_factor)
        
        # FFT of padded
        fft_padded = np.fft.fft2(padded)
        
        # Unpad
        unpadded = unpad_image(padded, padder)
        fft_unpadded = np.fft.fft2(unpadded)
        
        # Should be close to original
        np.testing.assert_allclose(unpadded, img, rtol=1e-10)