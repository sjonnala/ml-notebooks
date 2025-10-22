"""
Data processing utilities for machine learning projects.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path


class DataLoader:
    """Load and preprocess data from various sources."""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
    
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename: Name of the CSV file
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        file_path = self.data_path / filename
        return pd.read_csv(file_path, **kwargs)
    
    def load_excel(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            filename: Name of the Excel file
            **kwargs: Additional arguments for pandas.read_excel
            
        Returns:
            DataFrame with loaded data
        """
        file_path = self.data_path / filename
        return pd.read_excel(file_path, **kwargs)


class DataPreprocessor:
    """Preprocess data for machine learning."""
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame, 
        strategy: str = 'drop',
        columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'mode')
            columns: Specific columns to process (if None, process all)
            
        Returns:
            DataFrame with missing values handled
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.columns.tolist()
        
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=columns)
        elif strategy == 'mean':
            for col in columns:
                if df_copy[col].dtype in ['int64', 'float64']:
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in columns:
                if df_copy[col].dtype in ['int64', 'float64']:
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in columns:
                df_copy[col].fillna(df_copy[col].mode().iloc[0], inplace=True)
        
        return df_copy
    
    @staticmethod
    def split_features_target(
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataset into features and target.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y