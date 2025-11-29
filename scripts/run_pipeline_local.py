#!/usr/bin/env python3
"""
Run pipeline components locally to generate outputs for screenshots
This simulates the pipeline execution and generates metrics
"""

import sys
sys.path.insert(0, '.')

from src.pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)
import pandas as pd
import os

def run_pipeline_locally():
    """Run all pipeline components locally"""
    
    print("=" * 60)
    print("RUNNING PIPELINE COMPONENTS LOCALLY")
    print("=" * 60)
    
    # Step 1: Data Extraction
    print("\n[1/4] Extracting data from DVC...")
    try:
        extracted_path = data_extraction(
            dvc_remote_url='../../dvc-storage',
            data_path='data/raw/raw_data.csv',
            output_data_path='data/extracted/raw_data.csv'
        )
        print(f"   ✓ Data extracted to: {extracted_path}")
    except Exception as e:
        print(f"   Using existing data file...")
        extracted_path = 'data/raw/raw_data.csv'
    
    # Step 2: Data Preprocessing
    print("\n[2/4] Preprocessing data...")
    prep_result = data_preprocessing(
        input_data_path=extracted_path,
        output_train_path='data/processed/train.csv',
        output_test_path='data/processed/test.csv',
        test_size=0.2,
        random_state=42,
        scale_features=True
    )
    print(f"   ✓ Training samples: {prep_result.train_size}")
    print(f"   ✓ Test samples: {prep_result.test_size}")
    print(f"   ✓ Features: {prep_result.num_features}")
    
    # Step 3: Model Training
    print("\n[3/4] Training Random Forest model...")
    model_path = model_training(
        train_data_path='data/processed/train.csv',
        model_output_path='models/random_forest_model.joblib',
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    print(f"   ✓ Model saved to: {model_path}")
    
    # Step 4: Model Evaluation
    print("\n[4/4] Evaluating model...")
    eval_result = model_evaluation(
        model_path='models/random_forest_model.joblib',
        test_data_path='data/processed/test.csv',
        metrics_output_path='metrics/evaluation_metrics.txt',
        threshold_r2=0.5
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 60)
    print(f"\nEvaluation Results:")
    print(f"  R2 Score: {eval_result.r2_score:.4f}")
    print(f"  RMSE: {eval_result.rmse:.4f}")
    print(f"  Meets Threshold: {eval_result.meets_threshold}")
    
    print(f"\nMetrics saved to: metrics/evaluation_metrics.txt")
    print("\n" + "=" * 60)
    print("SCREENSHOT #3: View metrics file:")
    print("=" * 60)
    print("cat metrics/evaluation_metrics.txt")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline_locally()

