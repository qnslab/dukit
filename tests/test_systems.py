# -*- coding: utf-8 -*-
"""Tests for system classes (dukit.systems)."""

import numpy as np
import pytest
from pathlib import Path

from dukit.systems import (
    System,
    MelbSystem,
    LVControl,
    PyControl,
    Zyla,
    CryoWidefield,
    LegacyCryoWidefield,
    Argus,
    LegacyArgus,
    PyCryoWidefield,
)


class TestSystem:
    """Test base System class."""

    def test_system_can_instantiate_with_pixel_size(self):
        """Test that System can be instantiated with pixel_size."""
        system = System(pixel_size=1e-6)
        assert system is not None
        assert system._pixel_size == 1e-6

    def test_system_raises_without_params(self):
        """Test that System raises ValueError without pixel_size or microscope params."""
        with pytest.raises(ValueError):
            System()

    def test_system_can_instantiate_with_microscope_params(self):
        """Test that System can be instantiated with microscope params."""
        system = System(
            pixel_size=None,  # Don't set pixel_size, use microscope params
            sensor_pixel_pitch=6.5e-6,
            obj_mag=40,
            obj_ref_focal_length=200e-3,
            camera_tube_lens=300e-3,
        )
        assert system is not None

    def test_system_subclass_can_instantiate(self):
        """Test that System subclass can be instantiated."""
        class TestSystem(System):
            def read_image(self, *args, **kwargs):
                pass
            
            def get_hardware_binning(self, data_path):
                return 1
            
            def read_sweep_arr(self, data_path):
                return np.array([1.0, 2.0, 3.0])
            
            def get_raw_pixel_size(self, data_path):
                return 1e-6
            
            def get_bias_field(self, data_path, auto_read=False):
                return False, (None, None, None)
        
        system = TestSystem(pixel_size=1e-6)
        assert system is not None


class TestMelbSystem:
    """Test Melbourne system class."""

    def test_melb_system_methods_exist(self):
        """Test that MelbSystem has required methods."""
        # Check class has required methods
        assert hasattr(MelbSystem, 'read_image')
        assert hasattr(MelbSystem, 'get_hardware_binning')
        assert hasattr(MelbSystem, 'read_sweep_arr')
        assert hasattr(MelbSystem, 'get_raw_pixel_size')
        assert hasattr(MelbSystem, 'get_bias_field')
        assert hasattr(MelbSystem, 'norm')


class TestLVControl:
    """Test LabVIEW control system."""

    def test_lv_control_methods_exist(self):
        """Test LVControl has required methods."""
        assert hasattr(LVControl, 'read_image')
        assert hasattr(LVControl, 'get_hardware_binning')
        assert hasattr(LVControl, 'read_sweep_arr')


class TestPyControl:
    """Test Python control system."""

    def test_py_control_methods_exist(self):
        """Test PyControl has required methods."""
        assert hasattr(PyControl, 'read_image')
        assert hasattr(PyControl, 'get_hardware_binning')
        assert hasattr(PyControl, 'read_sweep_arr')


class TestZyla:
    """Test Zyla camera system."""

    def test_zyla_inherits_correctly(self):
        """Test Zyla inherits from LVControl."""
        assert issubclass(Zyla, LVControl)
        
    def test_zyla_has_system_params(self):
        """Test Zyla has predefined system parameters."""
        assert Zyla._obj_mag == 4
        assert Zyla._obj_ref_focal_length == 200e-3
        assert Zyla._camera_tube_lens == 300e-3
        assert Zyla._sensor_pixel_pitch == 6.5e-6


class TestCryoSystems:
    """Test cryogenic system classes."""

    def test_cryo_widefield_params(self):
        """Test CryoWidefield has pixel_size defined."""
        assert CryoWidefield._pixel_size == 59.6e-9

    def test_legacy_cryo_widefield_params(self):
        """Test LegacyCryoWidefield has pixel_size defined."""
        assert LegacyCryoWidefield._pixel_size == 59.6e-9

    def test_py_cryo_widefield_params(self):
        """Test PyCryoWidefield has pixel_size defined."""
        assert PyCryoWidefield._pixel_size == 59.6e-9


class TestArgusSystems:
    """Test Argus system classes."""

    def test_argus_params(self):
        """Test Argus has system parameters defined."""
        assert Argus._obj_mag == 40
        assert Argus._obj_ref_focal_length == 200e-3
        assert Argus._camera_tube_lens == 300e-3
        assert Argus._sensor_pixel_pitch == 6.5e-6

    def test_legacy_argus_params(self):
        """Test LegacyArgus has system parameters defined."""
        assert LegacyArgus._obj_mag == 40
        assert LegacyArgus._obj_ref_focal_length == 200e-3
        assert LegacyArgus._camera_tube_lens == 300e-3
        assert LegacyArgus._sensor_pixel_pitch == 6.5e-6


class TestSystemMethods:
    """Test system methods."""

    def test_norm_div(self):
        """Test div normalization."""
        sig = np.array([2.0, 4.0, 6.0])
        ref = np.array([1.0, 2.0, 3.0])
        result = System.norm(sig, ref, norm='div')
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_norm_sub(self):
        """Test sub normalization."""
        sig = np.array([2.0, 4.0, 6.0])
        ref = np.array([1.0, 2.0, 3.0])
        result = System.norm(sig, ref, norm='sub')
        expected = 1.0 + (sig - ref) / (sig + ref)
        np.testing.assert_array_almost_equal(result, expected)

    def test_norm_invalid(self):
        """Test invalid norm raises error."""
        sig = np.array([1.0, 2.0, 3.0])
        ref = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            System.norm(sig, ref, norm='invalid')

    def test_system_inheritance(self):
        """Test that all systems inherit from System."""
        systems = [
            MelbSystem, LVControl, PyControl, Zyla,
            CryoWidefield, LegacyCryoWidefield, PyCryoWidefield,
            Argus, LegacyArgus
        ]
        
        for system_class in systems:
            assert issubclass(system_class, System)
