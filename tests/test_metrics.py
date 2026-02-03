"""Tests for drowsiness_detection.metrics module."""

import pytest
import numpy as np
from drowsiness_detection.metrics import (
    calc_classification_metrics,
    calc_mean_and_std_of_classification_metrics
)


class TestCalcClassificationMetrics:
    """Tests for calc_classification_metrics function."""
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_trues = [np.array([1, 1, 0, 0])]
        y_preds = [np.array([1, 1, 0, 0])]
        
        accs, recalls, precisions, aucs = calc_classification_metrics(y_trues, y_preds)
        
        assert accs[0] == 1.0, "Perfect predictions should give 100% accuracy"
        assert recalls[0] == 1.0
        assert precisions[0] == 1.0
    
    def test_threshold_binarization(self):
        """Test that threshold of 0.5 is applied."""
        y_trues = [np.array([0.6, 0.4])]  # Should become [1, 0]
        y_preds = [np.array([0.7, 0.3])]  # Should become [1, 0]
        
        accs, recalls, precisions, aucs = calc_classification_metrics(y_trues, y_preds)
        assert accs[0] == 1.0, "Threshold-binarized predictions should match"
    
    def test_multiple_sets(self):
        """Test with multiple prediction sets."""
        y_trues = [
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0])
        ]
        y_preds = [
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0])
        ]
        
        accs, recalls, precisions, aucs = calc_classification_metrics(y_trues, y_preds)
        
        assert len(accs) == 2
        assert all(acc == 1.0 for acc in accs)


class TestMeanAndStd:
    """Tests for calc_mean_and_std_of_classification_metrics function."""
    
    def test_mean_std_perfect(self):
        """Test mean and std with perfect predictions."""
        y_trues = [
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0])
        ]
        y_preds = [
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0])
        ]
        
        (mean_acc, std_acc), _, _, _ = calc_mean_and_std_of_classification_metrics(y_trues, y_preds)
        
        assert mean_acc == 1.0
        assert std_acc == 0.0
    
    def test_mean_std_shape(self):
        """Test output is correct shape."""
        y_trues = [np.array([1, 0, 1])]
        y_preds = [np.array([1, 0, 1])]
        
        result = calc_mean_and_std_of_classification_metrics(y_trues, y_preds)
        
        # Should return 4 tuples: (mean, std) for accuracy, recall, precision, auc
        assert len(result) == 4
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
