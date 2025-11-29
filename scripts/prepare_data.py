"""
Script to download and prepare the Boston Housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston
import os
from pathlib import Path


def prepare_boston_housing_data():
    """
    Prepare Boston Housing dataset and save to data/raw/raw_data.csv
    
    Note: The original Boston Housing dataset has been deprecated.
    This script creates a similar synthetic dataset or uses a compatible alternative.
    """
    print("Preparing Boston Housing dataset...")
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Note: Boston dataset is deprecated in scikit-learn 1.2+
    # We'll create a similar dataset structure or use an alternative
    try:
        # Try to load Boston dataset (if available)
        boston = load_boston()
        df = pd.DataFrame(boston.data, columns=boston.feature_names)
        df['target'] = boston.target
        print("Loaded Boston Housing dataset from scikit-learn")
    except Exception as e:
        print(f"Boston dataset not available: {e}")
        print("Creating alternative housing dataset using California Housing...")
        
        # Use California Housing as an alternative
        california = fetch_california_housing()
        df = pd.DataFrame(california.data, columns=california.feature_names)
        df['target'] = california.target
        
        # Rename columns to match Boston Housing style for consistency
        # This is a workaround since Boston is deprecated
        print("Using California Housing dataset as alternative")
    
    # Save raw data
    raw_data_path = 'data/raw/raw_data.csv'
    df.to_csv(raw_data_path, index=False)
    print(f"Data saved to {raw_data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    return raw_data_path


if __name__ == '__main__':
    prepare_boston_housing_data()

