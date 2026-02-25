# -*- coding: utf-8 -*-
"""Tests for polygon handling (dukit.polygon)."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from dukit.polygon import (
    Polygon,
    PolygonSelectionWidget,
    load_polygon_nodes,
    polygon_selector,
)


class TestPolygon:
    """Test Polygon class - minimal smoke tests for actual API."""

    def test_polygon_creation(self, sample_polygon_coords):
        """Test creating a polygon with y, x coordinates."""
        poly = Polygon(sample_polygon_coords[:, 0], sample_polygon_coords[:, 1])
        assert poly is not None
        vertices = poly.get_yx()
        assert vertices.shape == (5, 2)  # 4 corners + 1 (closed polygon)

    def test_polygon_get_nodes(self, sample_polygon_coords):
        """Test getting polygon nodes as list."""
        poly = Polygon(sample_polygon_coords[:, 0], sample_polygon_coords[:, 1])
        nodes = poly.get_nodes()
        assert isinstance(nodes, list)
        assert len(nodes) == 5  # closed polygon has duplicate first/last

    def test_polygon_is_inside(self, sample_polygon_coords):
        """Test point-in-polygon check."""
        poly = Polygon(sample_polygon_coords[:, 0], sample_polygon_coords[:, 1])
        
        # Point inside (square from (3,3) to (8,8))
        result = poly.is_inside(5.0, 5.0)
        assert result > 0  # inside
        
        # Point outside
        result = poly.is_inside(2.0, 2.0)
        assert result < 0  # outside
        
        # Point on edge
        result = poly.is_inside(3.0, 5.0)
        assert result == 0  # on edge

    def test_polygon_is_inside_array(self, sample_polygon_coords):
        """Test point-in-polygon with array of points."""
        poly = Polygon(sample_polygon_coords[:, 0], sample_polygon_coords[:, 1])
        
        ys = np.array([5.0, 2.0, 6.0, 10.0])
        xs = np.array([5.0, 2.0, 6.0, 10.0])
        
        results = poly.is_inside(ys, xs)
        assert results.shape == (4,)
        assert results[0] > 0  # inside
        assert results[1] < 0  # outside
        assert results[2] > 0  # inside
        assert results[3] < 0  # outside


class TestPolygonIO:
    """Test polygon I/O functions."""

    def test_load_polygon_nodes_json(self, temp_dir):
        """Test loading polygon nodes from JSON file."""
        import json
        
        nodes_file = temp_dir / "test_nodes.json"
        test_data = {
            "nodes": [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            ],
            "image_shape": [10, 10],
            "padder": [[0, 0], [0, 0]]
        }
        
        with open(nodes_file, 'w') as f:
            json.dump(test_data, f)
        
        nodes_list = load_polygon_nodes(nodes_file)
        
        assert len(nodes_list) == 1
        nodes = nodes_list[0]
        assert nodes.shape == (4, 2)

    def test_load_nonexistent(self):
        """Test loading non-existent file raises error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            load_polygon_nodes("nonexistent_file.json")


class TestPolygonSelectionWidget:
    """Minimal smoke tests for PolygonSelectionWidget."""

    def test_polygon_selector_function_exists(self):
        """Test polygon_selector function is callable."""
        assert callable(polygon_selector)
