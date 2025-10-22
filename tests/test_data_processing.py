"""
Unit tests for the data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Add the src directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data import DataLoader, DataPreprocessor


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader(self.temp_dir)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Save test CSV
        self.csv_path = Path(self.temp_dir) / 'test_data.csv'
        self.test_data.to_csv(self.csv_path, index=False)
    
    def test_load_csv(self):
        """Test CSV loading functionality."""
        loaded_data = self.data_loader.load_csv('test_data.csv')
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create test data with missing values
        self.test_data = pd.DataFrame({
            'numeric': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical': ['a', 'b', None, 'd', 'e'],
            'target': [0, 1, 1, 0, 1]
        })
    
    def test_handle_missing_values_drop(self):
        """Test dropping missing values."""
        result = self.preprocessor.handle_missing_values(self.test_data, strategy='drop')
        assert len(result) == 3  # Should have 3 rows after dropping NaN rows
        assert not result.isnull().any().any()  # No missing values
    
    def test_handle_missing_values_mean(self):
        """Test filling missing values with mean."""
        result = self.preprocessor.handle_missing_values(
            self.test_data, 
            strategy='mean', 
            columns=['numeric']
        )
        assert not result['numeric'].isnull().any()  # No missing values in numeric column
        assert result['numeric'].loc[2] == 3.0  # Mean of [1, 2, 4, 5] is 3
    
    def test_split_features_target(self):
        """Test splitting features and target."""
        X, y = self.preprocessor.split_features_target(self.test_data, 'target')
        
        assert list(X.columns) == ['numeric', 'categorical']
        assert y.name == 'target'
        assert len(X) == len(y) == 5


if __name__ == '__main__':
    pytest.main([__file__])