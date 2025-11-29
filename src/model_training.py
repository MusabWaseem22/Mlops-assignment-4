"""
Model Training Script for Boston Housing Dataset
This script loads the processed data, trains a model, and saves it.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import argparse
import os
from pathlib import Path


def load_data(data_path: str) -> tuple:
    """
    Load the processed dataset.
    
    Args:
        data_path: Path to the processed CSV file
        
    Returns:
        Tuple of (X, y) - features and target
    """
    df = pd.read_csv(data_path)
    
    # Assuming the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    return X, y


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        random_state: Random state for reproducibility
        
    Returns:
        Trained model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train model on Boston Housing dataset')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to processed data CSV file')
    parser.add_argument('--model-path', type=str, default='models/model.joblib',
                        help='Path to save the trained model')
    parser.add_argument('--metrics-path', type=str, default='models/metrics.txt',
                        help='Path to save evaluation metrics')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of estimators for Random Forest')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    X, y = load_data(args.data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train, 
                       n_estimators=args.n_estimators,
                       random_state=args.random_state)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(model, args.model_path)
    print(f"\nModel saved to {args.model_path}")
    
    # Save metrics
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Metrics saved to {args.metrics_path}")
    
    return metrics


if __name__ == '__main__':
    main()

