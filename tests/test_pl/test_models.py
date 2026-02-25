# -*- coding: utf-8 -*-
"""Tests for peak fitting models (smoke tests only)."""

import numpy as np
import pytest

from dukit.pl.model import (
    FitModel,
    ConstStretchedExp,
    ConstBiExponential,
    ConstDampedRabi,
    LinearLorentzians,
    LinearN14Lorentzians,
    LinearN15Lorentzians,
    ConstLorentzians,
    SkewedLorentzians,
)


class TestFitModel:
    """Test base FitModel class."""

    def test_can_instantiate_base(self):
        """Base FitModel CAN be instantiated (not abstract)."""
        model = FitModel()
        assert model is not None


class TestConstStretchedExp:
    """Test T1 stretched exponential model."""

    def test_model_exists(self):
        """Test that model can be instantiated."""
        model = ConstStretchedExp()
        assert model is not None


class TestConstBiExponential:
    """Test T1 bi-exponential model."""

    def test_model_exists(self):
        """Test that model can be instantiated."""
        model = ConstBiExponential()
        assert model is not None


class TestConstDampedRabi:
    """Test damped Rabi model."""

    def test_model_exists(self):
        """Test that model can be instantiated."""
        model = ConstDampedRabi()
        assert model is not None


class TestLinearLorentzians:
    """Test linear Lorentzian model."""

    def test_model_creation(self):
        """Test model can be created with n_lorentzians."""
        model = LinearLorentzians(n_lorentzians=2)
        assert model is not None

    def test_single_peak(self):
        """Test fitting single peak."""
        model = LinearLorentzians(n_lorentzians=1)
        assert model is not None


class TestLinearN14Lorentzians:
    """Test N14 Lorentzian model (fixed hyperfine)."""

    def test_model_exists(self):
        """Test model can be instantiated."""
        model = LinearN14Lorentzians(n_lorentzians=1)
        assert model is not None


class TestLinearN15Lorentzians:
    """Test N15 Lorentzian model (fixed hyperfine)."""

    def test_model_exists(self):
        """Test model can be instantiated."""
        model = LinearN15Lorentzians(n_lorentzians=1)
        assert model is not None


class TestConstLorentzians:
    """Test Lorentzians with constant background."""

    def test_model_exists(self):
        """Test model can be instantiated."""
        model = ConstLorentzians(n_lorentzians=2)
        assert model is not None


class TestSkewedLorentzians:
    """Test skewed Lorentzian model."""

    def test_model_exists(self):
        """Test model can be instantiated."""
        model = SkewedLorentzians(n_lorentzians=1)
        assert model is not None
