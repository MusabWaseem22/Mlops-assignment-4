"""
Main Kubeflow Pipeline Definition
Orchestrates the ML workflow from data preprocessing to model evaluation.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, PipelineParam
from src.pipeline_components import (
    load_data_component,
    data_preprocessing,
    train_model_component,
    evaluate_model_component
)


@dsl.pipeline(
    name='boston-housing-ml-pipeline',
    description='A pipeline for training and evaluating a model on Boston Housing dataset'
)
def boston_housing_pipeline(
    input_data_path: str = 'data/raw/raw_data.csv',
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: int = 42,
    threshold_r2: float = 0.5
):
    """
    Main pipeline function that defines the ML workflow.
    
    Args:
        input_data_path: Path to the raw data CSV file
        test_size: Proportion of data for testing
        n_estimators: Number of trees in Random Forest
        random_state: Random state for reproducibility
        threshold_r2: Minimum R2 score threshold for model acceptance
    """
    
    # Step 1: Load Data
    load_op = load_data_component(input_data_path=input_data_path)
    
    # Step 2: Data Preprocessing
    preprocess_op = data_preprocessing(
        input_data=load_op.outputs['output_data'],
        test_size=test_size,
        random_state=random_state
    )
    preprocess_op.after(load_op)
    
    # Step 3: Train Model
    train_op = train_model_component(
        input_data=preprocess_op.outputs['output_data'],
        n_estimators=n_estimators,
        random_state=random_state
    )
    train_op.after(preprocess_op)
    
    # Step 4: Evaluate Model
    eval_op = evaluate_model_component(
        model=train_op.outputs['model'],
        input_data=preprocess_op.outputs['output_data'],
        threshold_r2=threshold_r2
    )
    eval_op.after(train_op)
    
    # Optional: Add conditional logic based on evaluation results
    # This would require additional components for model deployment


if __name__ == '__main__':
    # Compile the pipeline
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path='components/boston_housing_pipeline.yaml'
    )
    print("Pipeline compiled successfully!")

