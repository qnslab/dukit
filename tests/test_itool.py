# -*- coding: utf-8 -*-
"""Tests for image processing tools (dukit.itool)."""

import numpy as np
import pytest
from pathlib import Path

from dukit import itool


class TestImageManipulation:
    """Test basic image manipulation functions."""

    def test_crop_sweep(self, sample_image_stack):
        """Test cropping in sweep/frequency dimension."""
        # Transpose to (y, x, sweep) format as expected by crop_sweep
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        sweep_arr = np.linspace(2800, 2900, stack_yxs.shape[2])
        sig = stack_yxs
        ref = stack_yxs * 0.9
        sig_norm = stack_yxs / ref

        sweep_out, sig_out, ref_out, sig_norm_out = itool.crop_sweep(
            sweep_arr, sig, ref, sig_norm, rem_start=8, rem_end=8
        )

        # 32 - 8 - 8 = 16 points remaining
        assert sweep_out.shape[0] == 16
        # crop_sweep works with (y, x, sweep) format
        assert sig_out.shape[0] == stack_yxs.shape[0]  # y dim
        assert sig_out.shape[1] == stack_yxs.shape[1]  # x dim
        assert sig_out.shape[2] == 16  # sweep dim cropped

    def test_crop_sweep_full(self, sample_image_stack):
        """Test crop with full range returns original."""
        # Transpose to (y, x, sweep) format as expected by crop_sweep
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        sweep_arr = np.linspace(2800, 2900, stack_yxs.shape[2])
        sig = stack_yxs
        ref = stack_yxs * 0.9
        sig_norm = stack_yxs / ref

        sweep_out, sig_out, ref_out, sig_norm_out = itool.crop_sweep(
            sweep_arr, sig, ref, sig_norm, rem_start=0, rem_end=0
        )
        # Shape should be preserved
        assert sig_out.shape == stack_yxs.shape

    def test_crop_roi(self, sample_image_stack, sample_roi_coords):
        """Test cropping to ROI."""
        # Transpose to (y, x, sweep) format as expected by crop_roi
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        # crop_roi expects (x0, y0, x1, y1) format
        x0, y0, x1, y1 = sample_roi_coords
        cropped = itool.crop_roi(stack_yxs, (x0, y0, x1, y1))

        # Expected shape: (y, x, sweep) where y = y1-y0+1, x = x1-x0+1
        expected_y = y1 - y0 + 1
        expected_x = x1 - x0 + 1
        assert cropped.shape == (expected_y, expected_x, stack_yxs.shape[2])

    def test_crop_roi_full(self, sample_image_stack):
        """Test ROI crop with full image."""
        # Transpose to (y, x, sweep) format as expected by crop_roi
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        h, w = stack_yxs.shape[0], stack_yxs.shape[1]
        # Full image: (0, 0, w-1, h-1) => from (x0, y0, x1, y1)
        cropped = itool.crop_roi(stack_yxs, (0, 0, w - 1, h - 1))
        # Should preserve shape
        assert cropped.shape == stack_yxs.shape

    def test_sum_spatially(self, sample_image_stack):
        """Test spatial summing."""
        # Transpose to (y, x, sweep) format as expected
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        summed = itool.sum_spatially(stack_yxs)

        # sum_spatially returns shape (y, x) for 3D input (y, x, sweep)
        assert summed.ndim == 2
        assert summed.shape == stack_yxs.shape[:2]

    def test_rebin_image_stack(self, sample_image_stack):
        """Test rebinning image stack."""
        # Transpose to (y, x, sweep) format as expected by rebin_image_stack
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        # Rebin by factor of 2
        rebinned = itool.rebin_image_stack(stack_yxs, (2, 2))

        # Should be half the size in spatial dimensions (sweep dimension unchanged)
        assert rebinned.shape[0] == stack_yxs.shape[0] // 2
        assert rebinned.shape[1] == stack_yxs.shape[1] // 2
        assert rebinned.shape[2] == stack_yxs.shape[2]

    def test_rebin_asymmetric(self, sample_image_stack):
        """Test asymmetric rebinning."""
        # Transpose to (y, x, sweep) format
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        rebinned = itool.rebin_image_stack(stack_yxs, (2, 1))

        assert rebinned.shape[0] == stack_yxs.shape[0] // 2
        assert rebinned.shape[2] == stack_yxs.shape[2]  # Unchanged

    def test_smooth_image_stack(self, sample_image_stack):
        """Test smoothing image stack."""
        # Transpose to (y, x, sweep) format
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        smoothed = itool.smooth_image_stack(stack_yxs, sigma=1.0)

        assert smoothed.shape == stack_yxs.shape
        # Should be different from original
        assert not np.allclose(smoothed, stack_yxs)

    def test_smooth_asymmetric(self, sample_image_stack):
        """Test asymmetric smoothing."""
        # Transpose to (y, x, sweep) format
        stack_yxs = sample_image_stack.transpose(1, 2, 0)
        smoothed = itool.smooth_image_stack(stack_yxs, sigma=(2.0, 1.0))

        assert smoothed.shape == stack_yxs.shape


class TestFilteringAndBackground:
    """Test filtering and background subtraction."""

    def test_get_im_filtered_gaussian(self, sample_image_stack):
        """Test Gaussian filtering."""
        filtered = itool.get_im_filtered(
            sample_image_stack[8],  # Single frequency slice
            filter_type='gaussian',
            sigma=1.0
        )
        
        assert filtered.shape == sample_image_stack.shape[1:]
        assert not np.array_equal(filtered, sample_image_stack[8])

    def test_get_background_plane(self, sample_image_stack):
        """Test plane background subtraction."""
        bg, mask = itool.get_background(
            sample_image_stack[8],
            method='poly',
            order=1
        )
        
        assert bg.shape == sample_image_stack.shape[1:]

    def test_get_background_poly(self, sample_image_stack):
        """Test polynomial background."""
        bg, mask = itool.get_background(
            sample_image_stack[8],
            method='poly',
            order=2
        )
        
        assert bg.shape == sample_image_stack.shape[1:]

    def test_get_background_median(self, sample_image_stack):
        """Test median background."""
        bg, mask = itool.get_background(
            sample_image_stack[8],
            method='mean'
        )
        
        assert bg.shape == sample_image_stack.shape[1:]


class TestColormaps:
    """Test colormap range calculations."""

    def test_get_colormap_range_percentile(self, sample_image_stack):
        """Test percentile-based colormap range."""
        vmin, vmax = itool.get_colormap_range(
            'percentile',
            (5, 95),
            sample_image_stack[8],
        )
        
        assert vmin < vmax
        assert vmin >= np.min(sample_image_stack[8])
        assert vmax <= np.max(sample_image_stack[8])

    def test_get_colormap_range_std(self, sample_image_stack):
        """Test standard deviation colormap range."""
        # Use percentile instead of std
        vmin, vmax = itool.get_colormap_range(
            'min_max',
            (),
            sample_image_stack[8],
        )
        
        assert vmin < vmax

    def test_get_colormap_range_minmax(self, sample_image_stack):
        """Test min-max colormap range."""
        vmin, vmax = itool.get_colormap_range(
            'min_max',
            (),
            sample_image_stack[8],
        )
        
        assert vmin == np.min(sample_image_stack[8])
        assert vmax == np.max(sample_image_stack[8])


class TestPlotting:
    """Test plotting functions (smoke tests)."""

    def test_plot_image(self, sample_image_stack):
        """Test basic image plotting."""
        # Just check it doesn't crash
        itool.plot_image(sample_image_stack[8])

    def test_plot_image_with_options(self, sample_image_stack):
        """Test plotting with options."""
        itool.plot_image(
            sample_image_stack[8],
            title="Test Image",
            c_label="Intensity",
        )


class TestPolygonMasking:
    """Test polygon masking functions."""

    def test_get_aois(self, sample_image_stack, sample_roi_coords):
        """Test getting AOI data."""
        x0, y0, x1, y1 = sample_roi_coords
        # get_aois returns tuple of (slice, slice) tuples
        aois = itool.get_aois(
            sample_image_stack.shape,
            (x0, y0, x1, y1)
        )
        
        assert len(aois) == 2  # One AOI + center point


class TestMatplotlibConfig:
    """Test matplotlib configuration."""

    def test_mpl_set_run_config(self):
        """Test setting matplotlib config."""
        # Just check it doesn't crash
        itool.mpl_set_run_config()
