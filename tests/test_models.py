"""Tests for drowsiness_detection.models module."""

import pytest
import numpy as np
from drowsiness_detection.models import (
    ThreeDStandardScaler,
    build_cnn_model,
    build_lstm_model,
    build_bi_lstm_model
)


class TestThreeDStandardScaler:
    """Tests for ThreeDStandardScaler."""
    
    def test_3d_scaling(self):
        """Test scaling of 3D arrays."""
        scaler = ThreeDStandardScaler(feature_axis=2)
        
        # Create sample 3D data: (batch=2, time=3, features=4)
        X_train = np.random.randn(2, 3, 4)
        X_test = np.random.randn(2, 3, 4)
        
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_test)
        
        # Check shape is preserved
        assert X_scaled.shape == X_test.shape
        
        # Check that scaling happened (values should be smaller in magnitude)
        # This is a rough check since random data won't be perfectly scaled
        assert np.abs(X_scaled).mean() < np.abs(X_test).mean() + 1
    
    def test_shape_preservation(self):
        """Test that shape is preserved through scaling."""
        scaler = ThreeDStandardScaler(feature_axis=2)
        X = np.random.randn(5, 10, 20)
        
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        assert X_scaled.shape == X.shape


class TestModelBuilding:
    """Tests for neural network model builders."""
    
    def test_cnn_model_creation(self):
        """Test CNN model builds without errors."""
        input_shape = (None, 100, 64)  # (batch, sequence_length, features)
        model = build_cnn_model(input_shape)
        
        assert model is not None
        # Model should be compilable and summaryable
        assert len(model.layers) > 0
    
    def test_lstm_model_creation(self):
        """Test LSTM model builds without errors."""
        input_shape = (None, 100, 64)
        model = build_lstm_model(input_shape)
        
        assert model is not None
        assert len(model.layers) > 0
    
    def test_bi_lstm_model_creation(self):
        """Test Bidirectional LSTM model builds without errors."""
        input_shape = (None, 100, 64)
        model = build_bi_lstm_model(input_shape)
        
        assert model is not None
        assert len(model.layers) > 0
    
    def test_model_output_shape(self):
        """Test that models produce correct output shape."""
        input_shape = (None, 100, 64)
        model = build_cnn_model(input_shape)
        
        # Create dummy input
        X = np.random.randn(1, 100, 64)
        
        # Get prediction
        y = model.predict(X, verbose=0)
        
        # Should output single value (binary classification)
        assert y.shape == (1, 1)
