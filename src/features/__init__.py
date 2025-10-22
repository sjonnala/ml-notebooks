"""
Feature engineering utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


class FeatureScaler:
    """Scale features for machine learning."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize FeatureScaler.
        
        Args:
            method: Scaling method ('standard' or 'minmax')
        """
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform features.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        scaled_data = self.scaler.fit_transform(X)
        self.is_fitted = True
        return pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        
        scaled_data = self.scaler.transform(X)
        return pd.DataFrame(scaled_data, columns=X.columns, index=X.index)


class FeatureEncoder:
    """Encode categorical features."""
    
    @staticmethod
    def label_encode(
        df: pd.DataFrame, 
        columns: Union[str, List[str]]
    ) -> pd.DataFrame:
        """
        Apply label encoding to categorical columns.
        
        Args:
            df: Input DataFrame
            columns: Column(s) to encode
            
        Returns:
            DataFrame with encoded columns
        """
        df_copy = df.copy()
        
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        
        return df_copy
    
    @staticmethod
    def one_hot_encode(
        df: pd.DataFrame, 
        columns: Union[str, List[str]],
        drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns.
        
        Args:
            df: Input DataFrame
            columns: Column(s) to encode
            drop_first: Whether to drop the first category to avoid multicollinearity
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        df_copy = df.copy()
        
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col in df_copy.columns:
                dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=drop_first)
                df_copy = df_copy.drop(columns=[col])
                df_copy = pd.concat([df_copy, dummies], axis=1)
        
        return df_copy


class FeatureSelector:
    """Select important features for machine learning."""
    
    def __init__(self, method: str = 'univariate', k: int = 10):
        """
        Initialize FeatureSelector.
        
        Args:
            method: Feature selection method ('univariate')
            k: Number of top features to select
        """
        self.method = method
        self.k = k
        self.selector = None
        self.is_fitted = False
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        problem_type: str = 'classification'
    ) -> pd.DataFrame:
        """
        Fit selector and transform features.
        
        Args:
            X: Input features
            y: Target variable
            problem_type: Type of problem ('classification' or 'regression')
            
        Returns:
            Selected features
        """
        if self.method == 'univariate':
            if problem_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
            
            self.selector = SelectKBest(score_func=score_func, k=self.k)
        
        selected_data = self.selector.fit_transform(X, y)
        selected_columns = X.columns[self.selector.get_support()].tolist()
        self.is_fitted = True
        
        return pd.DataFrame(selected_data, columns=selected_columns, index=X.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted selector.
        
        Args:
            X: Input features
            
        Returns:
            Selected features
        """
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before transforming")
        
        selected_data = self.selector.transform(X)
        selected_columns = X.columns[self.selector.get_support()].tolist()
        
        return pd.DataFrame(selected_data, columns=selected_columns, index=X.index)