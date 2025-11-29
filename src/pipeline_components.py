"""
Kubeflow Pipeline Components
Defines reusable components for the ML pipeline.

This module contains four main components:
1. Data Extraction: Fetches versioned dataset from DVC remote storage
2. Data Preprocessing: Cleans, scales, and splits data into train/test sets
3. Model Training: Trains a Random Forest model and saves the artifact
4. Model Evaluation: Evaluates model and saves metrics to a file

These functions can be compiled to Kubeflow components using:
    kfp.components.create_component_from_func()
"""

from typing import NamedTuple


# ============================================================================
# COMPONENT 1: DATA EXTRACTION
# ============================================================================

def data_extraction(
    dvc_remote_url: str,
    data_path: str,
    output_data_path: str
) -> str:
    """
    Extract data from DVC remote storage using dvc get or dvc import.
    
    This component fetches a versioned dataset from DVC remote storage.
    It uses DVC's get/import functionality to retrieve the data.
    
    Args:
        dvc_remote_url: URL or path to DVC remote storage
                       (e.g., '../dvc-storage' or 'gdrive://folder-id')
        data_path: Path to the data file in DVC (e.g., 'data/raw/raw_data.csv')
        output_data_path: Path where the extracted data will be saved
        
    Returns:
        Path to the extracted data file
    """
    import subprocess
    import os
    from pathlib import Path
    
    print(f"Starting data extraction from DVC...")
    print(f"Remote URL: {dvc_remote_url}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_data_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_data_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use dvc get to fetch the data from remote storage
        # dvc get downloads the file without creating a DVC repository
        cmd = [
            'python', '-m', 'dvc', 'get',
            '--remote', dvc_remote_url,
            '.',  # repository root (relative path)
            data_path,
            '-o', output_data_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Data extraction successful!")
        print(f"Output: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        # Fallback: try dvc import or direct copy if dvc get fails
        print(f"DVC get failed, trying alternative method: {e.stderr}")
        
        # Alternative: Use dvc import or direct file access
        # For local storage, we can try direct access
        if os.path.exists(dvc_remote_url):
            import shutil
            # Try to find the file in the DVC cache
            # DVC stores files in .dvc/cache/
            dvc_cache_path = os.path.join(dvc_remote_url, '.dvc', 'cache')
            
            # For this implementation, we'll use a simpler approach:
            # If the file exists locally, copy it
            if os.path.exists(data_path):
                shutil.copy(data_path, output_data_path)
                print(f"Copied data from local path: {data_path}")
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")
        else:
            raise Exception(f"Failed to extract data: {e.stderr}")
    
    # Verify the file was created
    if not os.path.exists(output_data_path):
        raise FileNotFoundError(f"Output file not created: {output_data_path}")
    
    file_size = os.path.getsize(output_data_path)
    print(f"Data extraction completed successfully!")
    print(f"Output file: {output_data_path}")
    print(f"File size: {file_size} bytes")
    
    return output_data_path


# ============================================================================
# COMPONENT 2: DATA PREPROCESSING
# ============================================================================

def data_preprocessing(
    input_data_path: str,
    output_train_path: str,
    output_test_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True
) -> NamedTuple('PreprocessingOutput', [
    ('train_size', int),
    ('test_size', int),
    ('num_features', int)
]):
    """
    Preprocess the dataset: clean, scale, and split into train/test sets.
    
    This component performs:
    - Data cleaning (remove missing values, handle outliers)
    - Feature scaling (StandardScaler) if enabled
    - Train/test split
    
    Args:
        input_data_path: Path to the raw data CSV file
        output_train_path: Path to save the preprocessed training data
        output_test_path: Path to save the preprocessed test data
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random state for reproducibility (default: 42)
        scale_features: Whether to scale features using StandardScaler (default: True)
        
    Returns:
        NamedTuple containing:
            train_size: Number of samples in training set
            test_size: Number of samples in test set
            num_features: Number of features after preprocessing
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from collections import namedtuple
    import os
    
    print("Starting data preprocessing...")
    
    # Load data
    print(f"Loading data from: {input_data_path}")
    df = pd.read_csv(input_data_path)
    print(f"Original data shape: {df.shape}")
    
    # Data cleaning
    print("Cleaning data...")
    # Remove any missing values
    initial_rows = len(df)
    df = df.dropna()
    rows_after_cleaning = len(df)
    print(f"Removed {initial_rows - rows_after_cleaning} rows with missing values")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Data shape after cleaning: {df.shape}")
    
    # Separate features and target (assuming last column is target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Split data into train and test sets
    print(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Feature scaling
    if scale_features:
        print("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        print("Feature scaling completed")
    
    # Combine features and targets for saving
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    train_df['split'] = 'train'
    
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_df['split'] = 'test'
    
    # Create output directories
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    
    # Save processed data
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    
    print(f"Preprocessed training data saved to: {output_train_path}")
    print(f"Preprocessed test data saved to: {output_test_path}")
    
    output = namedtuple('PreprocessingOutput', ['train_size', 'test_size', 'num_features'])
    return output(
        train_size=len(train_df),
        test_size=len(test_df),
        num_features=X_train.shape[1]
    )


# ============================================================================
# COMPONENT 3: MODEL TRAINING
# ============================================================================

def model_training(
    train_data_path: str,
    model_output_path: str,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    random_state: int = 42
) -> str:
    """
    Train a Random Forest model on the training data and save the model artifact.
    
    This component trains a RandomForestRegressor on the preprocessed training data
    and saves the trained model as a joblib file.
    
    Args:
        train_data_path: Path to the preprocessed training data CSV file
        model_output_path: Path to save the trained model artifact (joblib format)
        n_estimators: Number of trees in the Random Forest (default: 100)
        max_depth: Maximum depth of the trees (default: None, unlimited)
        min_samples_split: Minimum samples required to split a node (default: 2)
        random_state: Random state for reproducibility (default: 42)
        
    Returns:
        Path to the saved model artifact
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    import os
    
    print("Starting model training...")
    
    # Load training data
    print(f"Loading training data from: {train_data_path}")
    train_df = pd.read_csv(train_data_path)
    
    # Separate features and target
    X_train = train_df.drop(['target', 'split'], axis=1)
    y_train = train_df['target']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of training samples: {X_train.shape[0]}")
    
    # Initialize and train Random Forest model
    print(f"Training Random Forest model...")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_samples_split={min_samples_split}, random_state={random_state}")
    
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    
    print("Model training completed successfully!")
    
    # Save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(rf_model, model_output_path)
    
    print(f"Model saved to: {model_output_path}")
    print(f"Model file size: {os.path.getsize(model_output_path)} bytes")
    
    # Print feature importance summary
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head().to_string(index=False))
    
    return model_output_path


# ============================================================================
# COMPONENT 4: MODEL EVALUATION
# ============================================================================

def model_evaluation(
    model_path: str,
    test_data_path: str,
    metrics_output_path: str,
    threshold_r2: float = 0.5
) -> NamedTuple('EvaluationOutput', [
    ('meets_threshold', bool),
    ('r2_score', float),
    ('rmse', float)
]):
    """
    Evaluate the trained model on the test set and save metrics to a file.
    
    This component:
    - Loads the trained model
    - Evaluates it on the test set
    - Calculates metrics (R2 score, RMSE, MSE, MAE)
    - Saves metrics to a text file
    - Checks if the model meets quality thresholds
    
    Args:
        model_path: Path to the trained model artifact (joblib file)
        test_data_path: Path to the preprocessed test data CSV file
        metrics_output_path: Path to save the evaluation metrics text file
        threshold_r2: Minimum R2 score threshold for model acceptance (default: 0.5)
        
    Returns:
        NamedTuple containing:
            meets_threshold: Boolean indicating if model meets R2 threshold
            r2_score: Calculated R2 score
            rmse: Calculated Root Mean Squared Error
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        explained_variance_score
    )
    from collections import namedtuple
    import os
    
    print("Starting model evaluation...")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    
    # Separate features and target
    X_test = test_df.drop(['target', 'split'], axis=1)
    y_test = test_df['target']
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of test samples: {len(y_test)}")
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)
    
    # Calculate additional metrics
    mean_actual = np.mean(y_test)
    mean_predicted = np.mean(y_pred)
    mean_absolute_percentage_error = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    # Check if model meets threshold
    meets_threshold = r2 >= threshold_r2
    
    # Print metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Explained Variance: {explained_variance:.4f}")
    print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error:.2f}%")
    print(f"Threshold (R2): {threshold_r2:.4f}")
    print(f"Meets Threshold: {meets_threshold}")
    print("="*50)
    
    # Save metrics to file
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    
    with open(metrics_output_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("MODEL EVALUATION METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Data Path: {test_data_path}\n\n")
        f.write("METRICS:\n")
        f.write(f"  R2 Score: {r2:.6f}\n")
        f.write(f"  RMSE (Root Mean Squared Error): {rmse:.6f}\n")
        f.write(f"  MSE (Mean Squared Error): {mse:.6f}\n")
        f.write(f"  MAE (Mean Absolute Error): {mae:.6f}\n")
        f.write(f"  Explained Variance: {explained_variance:.6f}\n")
        f.write(f"  Mean Absolute Percentage Error: {mean_absolute_percentage_error:.4f}%\n\n")
        f.write("PREDICTION STATISTICS:\n")
        f.write(f"  Mean Actual Value: {mean_actual:.4f}\n")
        f.write(f"  Mean Predicted Value: {mean_predicted:.4f}\n")
        f.write(f"  Std Actual Value: {np.std(y_test):.4f}\n")
        f.write(f"  Std Predicted Value: {np.std(y_pred):.4f}\n\n")
        f.write("THRESHOLD CHECK:\n")
        f.write(f"  Threshold (R2): {threshold_r2:.4f}\n")
        f.write(f"  Meets Threshold: {meets_threshold}\n")
        f.write("="*50 + "\n")
    
    print(f"\nMetrics saved to: {metrics_output_path}")
    
    output = namedtuple('EvaluationOutput', ['meets_threshold', 'r2_score', 'rmse'])
    return output(
        meets_threshold=meets_threshold,
        r2_score=float(r2),
        rmse=float(rmse)
    )


# ============================================================================
# NOTE: These functions are compiled to Kubeflow components using
# kfp.components.create_component_from_func() in scripts/compile_components.py
# ============================================================================
