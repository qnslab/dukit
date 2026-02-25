# -*- coding: utf-8 -*-
"""Tests for CPUFit and GPUFit backends."""

import numpy as np
import pytest

# Optional imports
try:
    from dukit.pl.cpufit import CPULorentzianFitter
    HAS_CPUFIT = True
except ImportError:
    HAS_CPUFIT = False
    CPULorentzianFitter = None

try:
    from dukit.pl.gpufit import GPULorentzianFitter
    HAS_GPUFIT = True
except ImportError:
    HAS_GPUFIT = False
    GPULorentzianFitter = None


class TestCPULorentzianFitter:
    """Test CPU fitting backend."""

    @pytest.mark.skipif(not HAS_CPUFIT, reason="cpufit not available")
    def test_fitter_exists(self):
        """Test that fitter can be instantiated."""
        fitter = CPULorentzianFitter()
        assert fitter is not None

    @pytest.mark.skipif(not HAS_CPUFIT, reason="cpufit not available")
    def test_single_peak_fit(self, sample_image_stack):
        """Test fitting single peak."""
        fitter = CPULorentzianFitter()
        freq_axis = np.linspace(2850, 2890, sample_image_stack.shape[0])
        
        # Use single pixel for speed
        data = sample_image_stack[:, 8, 8]
        
        # Initial guess
        guess = [0.3, 2870.0, 5.0, 1.0]  # amp, center, width, offset
        
        result = fitter.fit(freq_axis, data, guess)
        assert len(result) == 4  # Should return 4 parameters

    @pytest.mark.skipif(not HAS_CPUFIT, reason="cpufit not available")
    def test_multi_peak_fit(self, sample_image_stack):
        """Test fitting multiple peaks."""
        fitter = CPULorentzianFitter()
        freq_axis = np.linspace(2850, 2890, sample_image_stack.shape[0])
        
        data = sample_image_stack[:, 8, 8]
        
        # Guess for 2 peaks
        guess = [0.2, 2865.0, 5.0, 0.2, 2875.0, 5.0, 1.0]
        
        result = fitter.fit(freq_axis, data, guess, n_lorentzians=2)
        assert len(result) == 7  # 2*3 + offset


class TestGPULorentzianFitter:
    """Test GPU fitting backend."""

    @pytest.mark.skipif(not HAS_GPUFIT, reason="gpufit not available")
    def test_fitter_exists(self):
        """Test that fitter can be instantiated."""
        fitter = GPULorentzianFitter()
        assert fitter is not None

    @pytest.mark.skipif(not HAS_GPUFIT, reason="gpufit not available")
    def test_single_peak_fit(self, sample_image_stack):
        """Test fitting single peak on GPU."""
        fitter = GPULorentzianFitter()
        freq_axis = np.linspace(2850, 2890, sample_image_stack.shape[0])
        
        data = sample_image_stack[:, 8, 8]
        guess = [0.3, 2870.0, 5.0, 1.0]
        
        result = fitter.fit(freq_axis, data, guess)
        assert len(result) == 4

    @pytest.mark.skipif(not (HAS_CPUFIT and HAS_GPUFIT), reason="cpufit or gpufit not available")
    def test_gpu_vs_cpu_consistency(self, sample_image_stack):
        """Test that GPU and CPU give similar results."""
        freq_axis = np.linspace(2850, 2890, sample_image_stack.shape[0])
        data = sample_image_stack[:, 8, 8]
        guess = [0.3, 2870.0, 5.0, 1.0]
        
        cpu_fitter = CPULorentzianFitter()
        gpu_fitter = GPULorentzianFitter()
        
        cpu_result = cpu_fitter.fit(freq_axis, data, guess)
        gpu_result = gpu_fitter.fit(freq_axis, data, guess)
        
        # Should be reasonably close
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-3)