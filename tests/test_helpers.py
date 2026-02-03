"""Tests for drowsiness_detection.helpers module."""

import pytest
import numpy as np
from drowsiness_detection.helpers import binarize, digitize, label_to_one_hot_like


class TestBinarize:
    """Tests for binarize function."""
    
    def test_binarize_above_threshold(self):
        """Test binarization of values above threshold."""
        arr = np.array([0.3, 0.5, 0.7, 0.9])
        result = binarize(arr, threshold=0.5)
        expected = np.array([0, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_binarize_all_below_threshold(self):
        """Test binarization when all values are below threshold."""
        arr = np.array([0.1, 0.2, 0.3])
        result = binarize(arr, threshold=0.5)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_binarize_all_above_threshold(self):
        """Test binarization when all values are above threshold."""
        arr = np.array([0.6, 0.7, 0.8])
        result = binarize(arr, threshold=0.5)
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result, expected)


class TestDigitize:
    """Tests for digitize function."""
    
    def test_digitize_basic(self):
        """Test basic digitization."""
        arr = np.array([1.2, 2.6, 5.1])
        result = digitize(arr, shift=0.5, label_start_index=1)
        # bins = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        # 1.2 -> 0 -> 1, 2.6 -> 2 -> 3, 5.1 -> 4 -> 5
        assert result[0] == 1
        assert result[2] == 5
    
    def test_digitize_shape(self):
        """Test that digitize preserves input shape."""
        arr = np.array([[1.2, 2.6], [5.1, 7.2]])
        result = digitize(arr, shift=0.5, label_start_index=1)
        assert result.shape == arr.shape


class TestLabelToOneHotLike:
    """Tests for label_to_one_hot_like function."""
    
    def test_one_hot_like_basic(self):
        """Test ordinal encoding."""
        arr = np.array([1, 2, 3, 4])
        result = label_to_one_hot_like(arr, k=5)
        
        # arr[0] = 1: one_hots[0, :1] = [0]
        # arr[1] = 2: one_hots[1, :2] = [1, 0]
        # arr[2] = 3: one_hots[2, :3] = [1, 1, 0]
        # arr[3] = 4: one_hots[3, :4] = [1, 1, 1, 0]
        
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result[0], [0, 0, 0, 0])
        np.testing.assert_array_equal(result[1], [1, 0, 0, 0])
        np.testing.assert_array_equal(result[2], [1, 1, 0, 0])
        np.testing.assert_array_equal(result[3], [1, 1, 1, 0])
    
    def test_one_hot_like_shape(self):
        """Test output shape."""
        arr = np.array([1, 2, 3])
        result = label_to_one_hot_like(arr, k=9)
        assert result.shape == (3, 8)
