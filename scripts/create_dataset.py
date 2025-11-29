#!/usr/bin/env python3
"""
Simple script to create Boston Housing-like dataset.
This creates a synthetic dataset with similar structure to Boston Housing.
"""

import pandas as pd
import numpy as np
import os

def create_boston_housing_dataset():
    """Create a synthetic Boston Housing-like dataset."""
    print("Creating Boston Housing dataset...")
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate synthetic data similar to Boston Housing structure
    # Boston Housing has 13 features + 1 target
    np.random.seed(42)
    n_samples = 506
    
    # Create feature names (matching Boston Housing structure)
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Generate synthetic features
    data = {}
    data['CRIM'] = np.random.exponential(3.6, n_samples)
    data['ZN'] = np.random.exponential(11.4, n_samples)
    data['INDUS'] = np.random.normal(11.1, 6.9, n_samples)
    data['CHAS'] = np.random.choice([0, 1], n_samples, p=[0.93, 0.07])
    data['NOX'] = np.random.normal(0.55, 0.12, n_samples)
    data['RM'] = np.random.normal(6.3, 0.7, n_samples)
    data['AGE'] = np.random.normal(68.6, 28.1, n_samples)
    data['DIS'] = np.random.exponential(3.8, n_samples)
    data['RAD'] = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 24], n_samples, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
    data['TAX'] = np.random.normal(408.2, 168.5, n_samples)
    data['PTRATIO'] = np.random.normal(18.5, 2.2, n_samples)
    data['B'] = np.random.normal(356.7, 91.3, n_samples)
    data['LSTAT'] = np.random.normal(12.7, 7.1, n_samples)
    
    # Create target based on features (regression relationship)
    # This creates a realistic target variable
    target = (
        30 - 0.1 * data['CRIM'] - 0.05 * data['INDUS'] + 
        5 * data['RM'] - 0.1 * data['AGE'] - 1.5 * data['DIS'] - 
        0.5 * data['NOX'] - 0.1 * data['TAX'] - 0.5 * data['PTRATIO'] - 
        0.5 * data['LSTAT'] + np.random.normal(0, 3, n_samples)
    )
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['target'] = target
    
    # Ensure all values are positive where needed
    df['CRIM'] = np.abs(df['CRIM'])
    df['ZN'] = np.abs(df['ZN'])
    df['INDUS'] = np.abs(df['INDUS'])
    df['NOX'] = np.clip(df['NOX'], 0.3, 1.0)
    df['RM'] = np.clip(df['RM'], 3.5, 9.0)
    df['AGE'] = np.clip(df['AGE'], 0, 100)
    df['DIS'] = np.abs(df['DIS'])
    df['B'] = np.abs(df['B'])
    df['LSTAT'] = np.clip(df['LSTAT'], 0, 40)
    df['target'] = np.clip(df['target'], 5, 50)
    
    # Save raw data
    raw_data_path = 'data/raw/raw_data.csv'
    df.to_csv(raw_data_path, index=False)
    print(f"Data saved to {raw_data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Target range: [{df['target'].min():.2f}, {df['target'].max():.2f}]")
    
    return raw_data_path

if __name__ == '__main__':
    create_boston_housing_dataset()

