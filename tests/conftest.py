# -*- coding: utf-8 -*-
"""Shared fixtures for all tests."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def small_image_shape():
    """Return a small image shape for fast tests."""
    return (16, 16)


@pytest.fixture
def medium_image_shape():
    """Return a medium image shape for more realistic tests."""
    return (64, 64)


@pytest.fixture
def sample_bias_field():
    """Return a sample bias field (mag_T, theta_deg, phi_deg)."""
    return (0.01, 45.0, 30.0)  # 100 G, 45 deg polar, 30 deg azimuthal


@pytest.fixture
def sample_bias_field_post_gslac():
    """Return a bias field past GSLAC (102.4 G)."""
    return (0.1025, 0.0, 0.0)  # Just past GSLAC


@pytest.fixture
def sample_bias_field_pre_gslac():
    """Return a bias field before GSLAC."""
    return (0.05, 0.0, 0.0)  # 50 G, well before GSLAC


@pytest.fixture
def simple_uniform_bxyz(small_image_shape):
    """Return simple uniform Bxyz fields."""
    shape = small_image_shape
    Bx = np.ones(shape) * 5.0   # 5 G in x
    By = np.ones(shape) * 3.0   # 3 G in y  
    Bz = np.ones(shape) * 10.0  # 10 G in z
    return Bx, By, Bz


@pytest.fixture
def gradient_bxyz(medium_image_shape):
    """Return Bxyz fields with linear gradients."""
    shape = medium_image_shape
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    X, Y = np.meshgrid(x, y)
    
    Bx = X * 10.0  # Gradient in x
    By = Y * 5.0   # Gradient in y
    Bz = np.ones(shape) * 10.0  # Uniform in z
    
    return Bx, By, Bz


@pytest.fixture
def sample_pixel_size():
    """Return a sample pixel size in meters."""
    return 1.0e-6  # 1 micron


@pytest.fixture
def nv_u_defects_100_110():
    """Return sample NV orientations for <100>_<110> diamond."""
    # These are the standard orientations for CVD diamond
    return np.array([
        [np.sqrt(2/3), 0.0, np.sqrt(1/3)],
        [-np.sqrt(2/3), 0.0, np.sqrt(1/3)],
        [0.0, np.sqrt(2/3), -np.sqrt(1/3)],
        [0.0, -np.sqrt(2/3), -np.sqrt(1/3)]
    ])


@pytest.fixture
def orthogonal_u_defects():
    """Return orthogonal unit vectors for testing (simpler than NV)."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])


@pytest.fixture
def sample_nvfrequencies_4peaks(small_image_shape):
    """Return 4 NV frequencies (transitions) at ZFS."""
    shape = small_image_shape
    # At zero field, all 4 NVs have same frequency
    return tuple(np.full(shape, 2870.0) for _ in range(4))


@pytest.fixture
def sample_nvfrequencies_8peaks(small_image_shape):
    """Return 8 NV frequencies (full spectrum with bias)."""
    shape = small_image_shape
    # With bias, each NV has 2 transitions
    freqs = []
    for i in range(4):
        # Lower and higher transition for each NV
        freqs.append(np.full(shape, 2860.0 - i * 2))  # Lower branch
        freqs.append(np.full(shape, 2880.0 + i * 2))  # Higher branch
    return tuple(freqs)


@pytest.fixture
def sample_image_stack(small_image_shape):
    """Return a sample image stack (frequency/sweep dimension)."""
    shape = small_image_shape
    n_freq = 32
    # Create synthetic ODMR data: Lorentzian dips
    freq_axis = np.linspace(2850, 2890, n_freq)
    # Create grid with sweep on last axis: (sweep, x, y)
    y, x = shape
    freq_grid = freq_axis[:, np.newaxis, np.newaxis]
    
    # Two Lorentzian dips
    center1, width1, depth1 = 2865, 5, 0.3
    center2, width2, depth2 = 2875, 5, 0.2
    
    lorentz1 = depth1 / (1 + ((freq_grid - center1) / width1) ** 2)
    lorentz2 = depth2 / (1 + ((freq_grid - center2) / width2) ** 2)
    
    # Add noise and baseline
    baseline = 1.0
    noise = np.random.RandomState(42).randn(n_freq, x, y) * 0.02
    
    image_stack = baseline - lorentz1 - lorentz2 + noise
    return image_stack


@pytest.fixture
def sample_sweep_params():
    """Return sample sweep parameters."""
    return {
        'start': 2850.0,
        'stop': 2890.0,
        'n_points': 32
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_roi_coords():
    """Return sample ROI coordinates (x0, y0, x1, y1)."""
    return (5, 5, 11, 11)  # Center of 16x16 image


@pytest.fixture
def sample_polygon_coords():
    """Return sample polygon coordinates."""
    return np.array([
        [3, 3],
        [8, 3],
        [8, 8],
        [3, 8]
    ])


@pytest.fixture
def sample_fit_params():
    """Return sample fit parameters."""
    return {
        'amplitude': 0.3,
        'center': 2870.0,
        'width': 5.0,
        'offset': 1.0
    }


@pytest.fixture
def sample_tau_axis():
    """Return sample tau axis for T1 measurements."""
    return np.logspace(-1, 3, 32)  # 0.1 to 1000 us


@pytest.fixture
def sample_t1_data(small_image_shape, sample_tau_axis):
    """Return sample T1 decay data."""
    n_tau = len(sample_tau_axis)
    tau_grid = sample_tau_axis[:, np.newaxis, np.newaxis]
    
    # Stretched exponential decay
    T1 = 500.0  # us
    beta = 0.8
    amplitude = 0.5
    offset = 0.1
    
    decay = offset + amplitude * np.exp(-(tau_grid / T1) ** beta)
    noise = np.random.RandomState(123).randn(n_tau, *small_image_shape) * 0.02
    
    return decay + noise
