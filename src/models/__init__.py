"""
Machine learning model implementations.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


class BaseModel:
    """Base class for machine learning models."""
    
    def __init__(self, model_type: str = 'classification'):
        """
        Initialize BaseModel.
        
        Args:
            model_type: Type of model ('classification' or 'regression')
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X: Features
            y: Target values
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True


class RandomForestModel(BaseModel):
    """Random Forest model for classification and regression."""
    
    def __init__(self, model_type: str = 'classification', **kwargs):
        """
        Initialize RandomForestModel.
        
        Args:
            model_type: Type of model ('classification' or 'regression')
            **kwargs: Additional parameters for RandomForest
        """
        super().__init__(model_type)
        
        if model_type == 'classification':
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == 'regression':
            self.model = RandomForestRegressor(**kwargs)
        else:
            raise ValueError("model_type must be 'classification' or 'regression'")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Random Forest model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        return dict(zip(
            self.model.feature_names_in_,
            self.model.feature_importances_
        ))


class LinearModel(BaseModel):
    """Linear model for classification and regression."""
    
    def __init__(self, model_type: str = 'classification', **kwargs):
        """
        Initialize LinearModel.
        
        Args:
            model_type: Type of model ('classification' or 'regression')
            **kwargs: Additional parameters for Linear model
        """
        super().__init__(model_type)
        
        if model_type == 'classification':
            self.model = LogisticRegression(**kwargs)
        elif model_type == 'regression':
            self.model = LinearRegression(**kwargs)
        else:
            raise ValueError("model_type must be 'classification' or 'regression'")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Linear model."""
        self.model.fit(X, y)
        self.is_fitted = True


class ModelEvaluator:
    """Evaluate machine learning models."""
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred)
        }
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }