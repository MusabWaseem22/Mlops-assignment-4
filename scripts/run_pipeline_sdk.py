#!/usr/bin/env python3
"""
Run pipeline using KFP SDK directly (alternative to UI)
This generates pipeline runs that can be visualized and screenshotted
"""

import kfp
from kfp import dsl
from pipeline import boston_housing_pipeline

def run_pipeline_with_sdk():
    """Run pipeline using KFP SDK - works without full KFP deployment"""
    
    print("=" * 60)
    print("RUNNING PIPELINE WITH KFP SDK")
    print("=" * 60)
    
    # Compile pipeline
    from kfp import compiler
    
    print("\n1. Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path='pipeline.yaml'
    )
    print("   Pipeline compiled to: pipeline.yaml")
    
    # Note: To actually run, you need KFP backend
    # But you can still visualize the pipeline structure
    
    print("\n2. Pipeline structure:")
    print("   - Extract Data from DVC")
    print("   - Preprocess Data")  
    print("   - Train Random Forest Model")
    print("   - Evaluate Model")
    
    print("\n" + "=" * 60)
    print("For Task 3 deliverables:")
    print("=" * 60)
    print("1. minikube status - Already have")
    print("2. Pipeline graph - Can generate from pipeline.yaml")
    print("3. Metrics - Can run components locally and screenshot")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline_with_sdk()

