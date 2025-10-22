"""
Utility functions for the machine learning project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union
from pathlib import Path


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)


def create_directory_structure(base_path: Union[str, Path]) -> None:
    """
    Create the standard directory structure for ML projects.
    
    Args:
        base_path: Base path for the project
    """
    base_path = Path(base_path)
    
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'models',
        'notebooks/exploratory',
        'notebooks/modeling',
        'notebooks/evaluation',
        'docs',
        'tests'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)


def load_sample_data(dataset_name: str = 'iris') -> Tuple[pd.DataFrame, str]:
    """
    Load sample datasets for testing and experimentation.
    
    Args:
        dataset_name: Name of the dataset ('iris', 'boston', 'wine')
        
    Returns:
        Tuple of (DataFrame, target_column_name)
    """
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    
    if dataset_name == 'iris':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target'
    
    elif dataset_name == 'wine':
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target'
    
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target'
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def plot_feature_distributions(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot distributions of features.
    
    Args:
        df: Input DataFrame
        columns: Columns to plot (if None, plot all numeric columns)
        figsize: Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(columns):
        if i < len(axes):
            sns.histplot(df[col], ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame, 
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: Input DataFrame
        figsize: Figure size
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=figsize)
    correlation_matrix = numeric_df.corr()
    
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def generate_model_report(
    model_name: str,
    metrics: dict,
    feature_importance: Optional[dict] = None
) -> str:
    """
    Generate a formatted model performance report.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        feature_importance: Dictionary of feature importance scores
        
    Returns:
        Formatted report string
    """
    report = f"\n{'='*50}\n"
    report += f"MODEL PERFORMANCE REPORT: {model_name.upper()}\n"
    report += f"{'='*50}\n\n"
    
    report += "METRICS:\n"
    report += "-" * 20 + "\n"
    for metric, value in metrics.items():
        report += f"{metric.upper()}: {value:.4f}\n"
    
    if feature_importance:
        report += "\nTOP 10 IMPORTANT FEATURES:\n"
        report += "-" * 30 + "\n"
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            report += f"{feature}: {importance:.4f}\n"
    
    report += f"\n{'='*50}\n"
    
    return report