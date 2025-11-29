# Task 2: Building Kubeflow Pipeline Components - Documentation

## Component Overview

This document describes the four main Kubeflow pipeline components and their inputs/outputs.

---

## 1. Data Extraction Component

**Function:** `data_extraction()`

**Purpose:** Fetches a versioned dataset from DVC remote storage using `dvc get` or `dvc import`.

### Inputs:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `dvc_remote_url` | str | URL or path to DVC remote storage (e.g., '../dvc-storage', 'gdrive://folder-id', 's3://bucket/path') | Required |
| `data_path` | str | Path to the data file in DVC repository (e.g., 'data/raw/raw_data.csv') | Required |
| `output_data_path` | str | Path where the extracted data will be saved | Required |

### Outputs:

| Output | Type | Description |
|--------|------|-------------|
| Return value | str | Path to the extracted data file |

### Example Usage:
```python
extracted_data = data_extraction(
    dvc_remote_url='../dvc-storage',
    data_path='data/raw/raw_data.csv',
    output_data_path='data/extracted/raw_data.csv'
)
```

---

## 2. Data Preprocessing Component

**Function:** `data_preprocessing()`

**Purpose:** Cleans, scales, and splits data into train/test sets.

### Inputs:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_data_path` | str | Path to the raw data CSV file | Required |
| `output_train_path` | str | Path to save the preprocessed training data | Required |
| `output_test_path` | str | Path to save the preprocessed test data | Required |
| `test_size` | float | Proportion of data for testing | 0.2 |
| `random_state` | int | Random state for reproducibility | 42 |
| `scale_features` | bool | Whether to scale features using StandardScaler | True |

### Outputs:

| Output | Type | Description |
|--------|------|-------------|
| Return value | NamedTuple | Contains: |
| - `train_size` | int | Number of samples in training set |
| - `test_size` | int | Number of samples in test set |
| - `num_features` | int | Number of features after preprocessing |

### Example Usage:
```python
preprocessing_result = data_preprocessing(
    input_data_path='data/raw/raw_data.csv',
    output_train_path='data/processed/train.csv',
    output_test_path='data/processed/test.csv',
    test_size=0.2,
    random_state=42,
    scale_features=True
)
print(f"Train size: {preprocessing_result.train_size}")
print(f"Test size: {preprocessing_result.test_size}")
```

---

## 3. Model Training Component ⭐

**Function:** `model_training()`

**Purpose:** Trains a Random Forest model on the training data and saves the model artifact.

### Inputs:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `train_data_path` | str | Path to the preprocessed training data CSV file | Required |
| `model_output_path` | str | Path to save the trained model artifact (joblib format) | Required |
| `n_estimators` | int | Number of trees in the Random Forest | 100 |
| `max_depth` | int | Maximum depth of the trees (None = unlimited) | None |
| `min_samples_split` | int | Minimum samples required to split a node | 2 |
| `random_state` | int | Random state for reproducibility | 42 |

### Outputs:

| Output | Type | Description |
|--------|------|-------------|
| Return value | str | Path to the saved model artifact |

### Component Behavior:

1. **Loads Training Data:** Reads the preprocessed training CSV file
2. **Separates Features and Target:** Assumes last columns are 'target' and 'split'
3. **Initializes Model:** Creates a RandomForestRegressor with specified parameters
4. **Trains Model:** Fits the model on the training data
5. **Saves Model:** Exports the trained model as a joblib file
6. **Reports Feature Importance:** Prints top 5 most important features

### Example Usage:
```python
model_path = model_training(
    train_data_path='data/processed/train.csv',
    model_output_path='models/random_forest_model.joblib',
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
```

### Model Training Parameters Explained:

- **n_estimators (100):** Number of decision trees in the forest. More trees = better performance but slower training.
- **max_depth (None):** Maximum depth of each tree. None means nodes are expanded until all leaves are pure.
- **min_samples_split (2):** Minimum number of samples required to split an internal node.
- **random_state (42):** Seed for random number generator to ensure reproducibility.

### Output Model Format:

The trained model is saved as a joblib file containing:
- The trained RandomForestRegressor object
- All hyperparameters used during training
- Feature names and structure

---

## 4. Model Evaluation Component

**Function:** `model_evaluation()`

**Purpose:** Evaluates the trained model on the test set and saves metrics to a file.

### Inputs:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_path` | str | Path to the trained model artifact (joblib file) | Required |
| `test_data_path` | str | Path to the preprocessed test data CSV file | Required |
| `metrics_output_path` | str | Path to save the evaluation metrics text file | Required |
| `threshold_r2` | float | Minimum R2 score threshold for model acceptance | 0.5 |

### Outputs:

| Output | Type | Description |
|--------|------|-------------|
| Return value | NamedTuple | Contains: |
| - `meets_threshold` | bool | Whether model meets R2 threshold |
| - `r2_score` | float | Calculated R2 score |
| - `rmse` | float | Calculated Root Mean Squared Error |

### Metrics Calculated:

1. **R2 Score:** Coefficient of determination (measures how well model explains variance)
2. **RMSE:** Root Mean Squared Error (average prediction error)
3. **MSE:** Mean Squared Error
4. **MAE:** Mean Absolute Error
5. **Explained Variance:** Proportion of variance explained by the model
6. **MAPE:** Mean Absolute Percentage Error

### Example Usage:
```python
evaluation_result = model_evaluation(
    model_path='models/random_forest_model.joblib',
    test_data_path='data/processed/test.csv',
    metrics_output_path='metrics/evaluation_metrics.txt',
    threshold_r2=0.5
)

if evaluation_result.meets_threshold:
    print(f"Model passed! R2 Score: {evaluation_result.r2_score:.4f}")
else:
    print(f"Model failed threshold. R2 Score: {evaluation_result.r2_score:.4f}")
```

### Metrics File Format:

The metrics are saved to a text file containing:
- All calculated metrics (R2, RMSE, MSE, MAE, etc.)
- Prediction statistics (mean, std of actual vs predicted)
- Threshold check results

---

## Component Compilation

All components are compiled to YAML files using:

```python
from kfp import components

yaml_content = components.create_component_from_func(
    func=component_function,
    base_image='python:3.9',
    packages_to_install=['required', 'packages']
)
```

### Compiling Components:

Run the compilation script:
```bash
python3 scripts/compile_components.py
```

This will generate YAML files in the `components/` directory:
- `data_extraction.yaml`
- `data_preprocessing.yaml`
- `model_training.yaml`
- `model_evaluation.yaml`

---

## Pipeline Workflow

The components are designed to work together in the following sequence:

```
Data Extraction → Data Preprocessing → Model Training → Model Evaluation
     ↓                    ↓                  ↓                 ↓
  raw_data.csv    train.csv + test.csv   model.joblib    metrics.txt
```

1. **Data Extraction:** Retrieves versioned data from DVC
2. **Data Preprocessing:** Cleans and prepares data for training
3. **Model Training:** Trains Random Forest model on training data
4. **Model Evaluation:** Evaluates model performance on test data

---

## Notes

- All components use Python 3.9 as the base image
- Components are self-contained and handle their own imports
- File paths should be absolute or relative to the component's working directory
- The Random Forest model is suitable for regression tasks (Boston Housing dataset)
- Metrics are saved to human-readable text files for easy review

