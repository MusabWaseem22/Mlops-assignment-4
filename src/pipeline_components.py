"""
Kubeflow Pipeline Components
Defines reusable components for the ML pipeline.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics
from typing import NamedTuple


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy', 'scikit-learn']
)
def load_data_component(
    input_data_path: str,
    output_data: Output[Dataset]
):
    """
    Load data from a file path and create a Dataset artifact.
    
    Args:
        input_data_path: Path to the raw data CSV file
        output_data: Output dataset artifact
    """
    import pandas as pd
    import shutil
    
    # Copy file to output location (KFP will handle the path)
    df = pd.read_csv(input_data_path)
    df.to_csv(output_data.path, index=False)
    print(f"Data loaded from {input_data_path}")


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy', 'scikit-learn']
)
def data_preprocessing(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42
) -> NamedTuple('PreprocessingOutput', [('train_size', int), ('test_size', int)]):
    """
    Preprocess the Boston Housing dataset.
    
    Args:
        input_data: Input dataset (raw data CSV)
        output_data: Output dataset (processed data CSV)
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple with train and test sizes
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from collections import namedtuple
    
    # Load data
    df = pd.read_csv(input_data.path)
    
    # Basic preprocessing
    # Remove any missing values
    df = df.dropna()
    
    # Separate features and target (assuming last column is target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Combine train and test sets with split indicator
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df['split'] = 'train'
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df['split'] = 'test'
    
    # Save processed data
    processed_df = pd.concat([train_df, test_df], ignore_index=True)
    processed_df.to_csv(output_data.path, index=False)
    
    output = namedtuple('PreprocessingOutput', ['train_size', 'test_size'])
    return output(train_size=len(train_df), test_size=len(test_df))


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'joblib']
)
def train_model_component(
    input_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
    random_state: int = 42
):
    """
    Train a Random Forest model on the processed data.
    
    Args:
        input_data: Processed dataset
        model: Output model artifact
        metrics: Output metrics artifact
        n_estimators: Number of trees in the forest
        random_state: Random state for reproducibility
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Load processed data
    df = pd.read_csv(input_data.path)
    
    # Split train and test
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    X_train = train_df.drop(['target', 'split'], axis=1)
    y_train = train_df['target']
    X_test = test_df.drop(['target', 'split'], axis=1)
    y_test = test_df['target']
    
    # Train model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    joblib.dump(rf_model, model.path)
    
    # Log metrics
    metrics.log_metric('mse', float(mse))
    metrics.log_metric('rmse', float(rmse))
    metrics.log_metric('r2_score', float(r2))
    
    print(f"Model trained successfully!")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['joblib', 'scikit-learn']
)
def evaluate_model_component(
    model: Input[Model],
    input_data: Input[Dataset],
    metrics: Output[Metrics],
    threshold_r2: float = 0.5
) -> NamedTuple('EvaluationOutput', [('meets_threshold', bool)]):
    """
    Evaluate the trained model and check if it meets quality thresholds.
    
    Args:
        model: Trained model artifact
        input_data: Test dataset
        metrics: Output metrics artifact
        threshold_r2: Minimum R2 score threshold
        
    Returns:
        Whether the model meets the threshold
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from collections import namedtuple
    
    # Load model and data
    rf_model = joblib.load(model.path)
    df = pd.read_csv(input_data.path)
    
    # Get test data
    test_df = df[df['split'] == 'test'].copy()
    X_test = test_df.drop(['target', 'split'], axis=1)
    y_test = test_df['target']
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log additional metrics
    metrics.log_metric('mse', float(mse))
    metrics.log_metric('rmse', float(rmse))
    metrics.log_metric('mae', float(mae))
    metrics.log_metric('r2_score', float(r2))
    
    # Check threshold
    meets_threshold = r2 >= threshold_r2
    
    output = namedtuple('EvaluationOutput', ['meets_threshold'])
    return output(meets_threshold=meets_threshold)

