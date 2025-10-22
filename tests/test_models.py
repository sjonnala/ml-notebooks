"""
Unit tests for the machine learning models.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Add the src directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from models import RandomForestModel, LinearModel, ModelEvaluator
except ImportError:
    # Skip tests if dependencies not available
    pytest.skip("Models module dependencies not available", allow_module_level=True)


class TestRandomForestModel:
    """Test cases for RandomForestModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y_classification = np.random.randint(0, 2, 100)
        self.y_regression = np.random.randn(100)
    
    def test_classification_model(self):
        """Test Random Forest classification."""
        model = RandomForestModel(model_type='classification', n_estimators=10, random_state=42)
        
        # Test fitting
        model.fit(self.X, self.y_classification)
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y_classification)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 3  # 3 features
        assert all(0 <= imp <= 1 for imp in importance.values())
    
    def test_regression_model(self):
        """Test Random Forest regression."""
        model = RandomForestModel(model_type='regression', n_estimators=10, random_state=42)
        
        # Test fitting
        model.fit(self.X, self.y_regression)
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y_regression)
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        
        # Create sample predictions
        self.y_true_class = np.array([0, 1, 1, 0, 1])
        self.y_pred_class = np.array([0, 1, 0, 0, 1])
        
        self.y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_reg = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    
    def test_evaluate_classification(self):
        """Test classification evaluation."""
        metrics = self.evaluator.evaluate_classification(self.y_true_class, self.y_pred_class)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['accuracy'] == 0.8  # 4 out of 5 correct
    
    def test_evaluate_regression(self):
        """Test regression evaluation."""
        metrics = self.evaluator.evaluate_regression(self.y_true_reg, self.y_pred_reg)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['rmse'] == np.sqrt(metrics['mse'])


if __name__ == '__main__':
    pytest.main([__file__])