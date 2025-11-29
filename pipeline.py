"""
Main Kubeflow Pipeline Definition
Orchestrates the ML workflow from data extraction to model evaluation.

This pipeline:
1. Extracts data from DVC remote storage
2. Preprocesses the data (cleaning, scaling, splitting)
3. Trains a Random Forest model
4. Evaluates the model and saves metrics
"""

from kfp import dsl, components
from kfp.dsl import PipelineTask
from typing import NamedTuple


# Load components from YAML files
data_extraction_component = components.load_component_from_file('components/data_extraction.yaml')
data_preprocessing_component = components.load_component_from_file('components/data_preprocessing.yaml')
model_training_component = components.load_component_from_file('components/model_training.yaml')
model_evaluation_component = components.load_component_from_file('components/model_evaluation.yaml')


@dsl.pipeline(
    name='boston-housing-ml-pipeline',
    description='A complete ML pipeline for training and evaluating a model on Boston Housing dataset. '
                'The pipeline includes: data extraction from DVC, preprocessing, model training, and evaluation.'
)
def boston_housing_pipeline(
    dvc_remote_url: str = '../../dvc-storage',
    dvc_data_path: str = 'data/raw/raw_data.csv',
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 0,  # 0 means None/unlimited
    min_samples_split: int = 2,
    threshold_r2: float = 0.5,
    scale_features: bool = True
):
    """
    Main pipeline function that orchestrates the complete ML workflow.
    
    Args:
        dvc_remote_url: URL or path to DVC remote storage
        dvc_data_path: Path to the data file in DVC repository
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random state for reproducibility (default: 42)
        n_estimators: Number of trees in Random Forest (default: 100)
        max_depth: Maximum depth of trees, 0 = unlimited (default: 0)
        min_samples_split: Minimum samples to split a node (default: 2)
        threshold_r2: Minimum R2 score threshold for model acceptance (default: 0.5)
        scale_features: Whether to scale features using StandardScaler (default: True)
    """
    
    # ========================================================================
    # STEP 1: Data Extraction
    # ========================================================================
    # Extract versioned dataset from DVC remote storage
    extract_task = data_extraction_component(
        dvc_remote_url=dvc_remote_url,
        data_path=dvc_data_path,
        output_data_path='data/extracted/raw_data.csv'
    )
    extract_task.set_display_name('1. Extract Data from DVC')
    
    # ========================================================================
    # STEP 2: Data Preprocessing
    # ========================================================================
    # Clean, scale, and split data into train/test sets
    preprocess_task = data_preprocessing_component(
        input_data_path=extract_task.output,
        output_train_path='data/processed/train.csv',
        output_test_path='data/processed/test.csv',
        test_size=test_size,
        random_state=random_state,
        scale_features=scale_features
    )
    preprocess_task.set_display_name('2. Preprocess Data')
    preprocess_task.after(extract_task)
    
    # ========================================================================
    # STEP 3: Model Training
    # ========================================================================
    # Train Random Forest model on preprocessed training data
    # Note: The training data path must match what preprocessing saved
    train_task = model_training_component(
        train_data_path='data/processed/train.csv',
        model_output_path='models/random_forest_model.joblib',
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    train_task.set_display_name('3. Train Random Forest Model')
    train_task.after(preprocess_task)
    
    # ========================================================================
    # STEP 4: Model Evaluation
    # ========================================================================
    # Evaluate the trained model on test set and save metrics
    # Note: Paths must match what previous components saved
    evaluate_task = model_evaluation_component(
        model_path=train_task.output,
        test_data_path='data/processed/test.csv',
        metrics_output_path='metrics/evaluation_metrics.txt',
        threshold_r2=threshold_r2
    )
    evaluate_task.set_display_name('4. Evaluate Model')
    evaluate_task.after(train_task)
    
    # Pipeline output summary
    # The pipeline completes successfully if all steps finish without errors


if __name__ == '__main__':
    # Compile the pipeline to YAML format
    from kfp import compiler
    
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path='pipeline.yaml'
    )
    print("Pipeline compiled successfully to: pipeline.yaml")
    print("\nNext steps:")
    print("1. Start Minikube and deploy Kubeflow Pipelines")
    print("2. Upload pipeline.yaml to Kubeflow Pipelines UI")
    print("3. Run the pipeline with appropriate parameters")
