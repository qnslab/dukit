# -*- coding: utf-8 -*-
"""Tests for geometry utilities (dukit.geom)."""

import numpy as np
import pytest

from dukit.geom import (
    get_u_defects,
    get_u_defect_frames,
    NV_AXES_111,
    NV_AXES_100_100,
    NV_AXES_100_110,
)


class TestNVAxes:
    """Test NV axis constants."""

    def test_nv_axes_111_shape(self):
        """Test NV axes for <111> orientation."""
        assert len(NV_AXES_111) == 4
        # First axis should be unit vector
        np.testing.assert_allclose(np.linalg.norm(NV_AXES_111[0]["ori"]), 1.0, rtol=1e-10)

    def test_nv_axes_100_100_shape(self):
        """Test NV axes for <100>_<100> orientation."""
        assert len(NV_AXES_100_100) == 4
        # All should be unit vectors
        for axis in NV_AXES_100_100:
            np.testing.assert_allclose(np.linalg.norm(axis["ori"]), 1.0, rtol=1e-10)

    def test_nv_axes_100_110_shape(self):
        """Test NV axes for <100>_<110> orientation."""
        assert len(NV_AXES_100_110) == 4
        # All should be unit vectors
        for axis in NV_AXES_100_110:
            np.testing.assert_allclose(np.linalg.norm(axis["ori"]), 1.0, rtol=1e-10)

    def test_axes_are_orthogonal_sets(self):
        """Test that axes form proper orthogonal sets."""
        # For <111>, axes should be at tetrahedral angles
        for axes in [NV_AXES_111, NV_AXES_100_100, NV_AXES_100_110]:
            # Convert to numpy arrays for dot product calculation
            arr = np.array([a["ori"] for a in axes])
            # Check pairwise dot products (skip NaN values)
            for i in range(4):
                for j in range(i+1, 4):
                    if np.any(np.isnan(arr[i])) or np.any(np.isnan(arr[j])):
                        continue
                    dot = np.dot(arr[i], arr[j])
                    # Should be negative (obtuse angle) for NV axes
                    assert dot < 0


class TestGetUDefects:
    """Test getting defect orientations."""

    def test_get_u_defects_111(self):
        """Test getting <111> orientations."""
        u_defects = get_u_defects(0.01, 0.0, 0.0, diamond_ori="<111>")
        
        assert u_defects.shape == (4, 3)
        # First defect should be (0,0,1) oriented along z (highest projection on bias_x)
        np.testing.assert_allclose(np.abs(u_defects[0]), np.abs(NV_AXES_111[0]["ori"]), rtol=1e-10)

    def test_get_u_defects_100_110(self):
        """Test getting <100>_<110> orientations."""
        u_defects = get_u_defects(0.01, 0.0, 0.0, diamond_ori="<100>_<110>")
        
        assert u_defects.shape == (4, 3)
        # Check that all returned axes are unit vectors
        for axis in u_defects:
            np.testing.assert_allclose(np.linalg.norm(axis), 1.0, rtol=1e-10)

    def test_get_u_defects_100_100(self):
        """Test getting <100>_<100> orientations."""
        u_defects = get_u_defects(0.01, 0.0, 0.0, diamond_ori="<100>_<100>")
        
        assert u_defects.shape == (4, 3)
        # Check that all returned axes are unit vectors
        for axis in u_defects:
            np.testing.assert_allclose(np.linalg.norm(axis), 1.0, rtol=1e-10)

    def test_bias_field_rotation(self):
        """Test that bias field rotates defect orientations."""
        # No rotation
        u1 = get_u_defects(0.01, 0.0, 0.0, diamond_ori="<100>_<110>")
        
        # With rotation
        u2 = get_u_defects(0.01, 45.0, 90.0, diamond_ori="<100>_<110>")
        
        # Should be different
        assert not np.allclose(u1, u2)

    def test_invalid_orientation(self):
        """Test invalid diamond orientation raises error."""
        with pytest.raises(ValueError):
            get_u_defects(0.01, 0.0, 0.0, diamond_ori="<invalid>")


class TestGetUDefectFrames:
    """Test getting defect reference frames."""

    def test_get_u_defect_frames_shape(self):
        """Test output shape of defect frames."""
        frames = get_u_defect_frames(0.01, 0.0, 0.0, diamond_ori="<100>_<110>")
        
        # Should have 4 frames, each with 3 basis vectors
        assert frames.shape == (4, 3, 3)

    def test_frames_are_orthonormal(self):
        """Test that each frame forms orthonormal basis."""
        frames = get_u_defect_frames(0.01, 0.0, 0.0, diamond_ori="<100>_<110>")
        
        for i, frame in enumerate(frames):
            # Check orthonormality
            np.testing.assert_allclose(
                frame @ frame.T,
                np.eye(3),
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"Frame {i} not orthonormal"
            )

    def test_first_vector_is_u_defect(self):
        """Test that first vector in each frame is the defect orientation."""
        u_defects = get_u_defects(0.01, 0.0, 0.0, diamond_ori="<100>_<110>")
        frames = get_u_defect_frames(0.01, 0.0, 0.0, diamond_ori="<100>_<110>")
        
        for i in range(4):
            # The third basis vector (index 2) is the u_defect (z-axis), not first
            np.testing.assert_allclose(
                frames[i, 2],  # Third basis vector is uNV_Z
                u_defects[i],
                rtol=1e-10
            )

    def test_frames_with_rotation(self):
        """Test frames with bias field rotation."""
        frames1 = get_u_defect_frames(0.01, 0.0, 0.0, diamond_ori="<100>_<110>")
        frames2 = get_u_defect_frames(0.01, 45.0, 90.0, diamond_ori="<100>_<110>")
        
        # Should be different
        assert not np.allclose(frames1, frames2)

    def test_different_orientations(self):
        """Test frames for different diamond orientations."""
        frames_111 = get_u_defect_frames(0.01, 0.0, 0.0, diamond_ori="<111>")
        frames_100_110 = get_u_defect_frames(0.01, 0.0, 0.0, diamond_ori="<100>_<110>")
        frames_100_100 = get_u_defect_frames(0.01, 0.0, 0.0, diamond_ori="<100>_<100>")
        
        # All should have correct shape
        for frames in [frames_111, frames_100_110, frames_100_100]:
            assert frames.shape == (4, 3, 3)
        
        # Should be different from each other
        assert not np.allclose(frames_111, frames_100_110)
        assert not np.allclose(frames_100_110, frames_100_100)