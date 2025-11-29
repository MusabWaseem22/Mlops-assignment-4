#!/usr/bin/env python3
"""
Script to compile Kubeflow Pipeline components to YAML files.

This script uses kfp.components.create_component_from_func to convert
Python functions into reusable Kubeflow components saved as YAML files.

Usage:
    python3 scripts/compile_components.py
    
Note: This requires kfp to be installed:
    pip install kfp>=2.4.0
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import components
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from kfp import components
except ImportError:
    print("ERROR: kfp package not found. Please install it first:")
    print("  pip install kfp>=2.4.0")
    sys.exit(1)

# Import the component functions
from src.pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)


def compile_all_components():
    """Compile all pipeline components to YAML files."""
    
    # Create components directory if it doesn't exist
    components_dir = Path(__file__).parent.parent / 'components'
    components_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("COMPILING KUBEFLOW PIPELINE COMPONENTS")
    print("=" * 60)
    
    # Component 1: Data Extraction
    print("\n1. Compiling Data Extraction component...")
    try:
        data_extraction_yaml = components.create_component_from_func(
            func=data_extraction,
            base_image='python:3.9',
            packages_to_install=['dvc[s3]>=3.0.0', 'pandas>=2.0.0']
        )
        
        output_path = components_dir / 'data_extraction.yaml'
        with open(output_path, 'w') as f:
            f.write(data_extraction_yaml)
        
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Component 2: Data Preprocessing
    print("\n2. Compiling Data Preprocessing component...")
    try:
        data_preprocessing_yaml = components.create_component_from_func(
            func=data_preprocessing,
            base_image='python:3.9',
            packages_to_install=['pandas>=2.0.0', 'numpy>=1.24.0', 'scikit-learn>=1.3.0']
        )
        
        output_path = components_dir / 'data_preprocessing.yaml'
        with open(output_path, 'w') as f:
            f.write(data_preprocessing_yaml)
        
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Component 3: Model Training
    print("\n3. Compiling Model Training component...")
    try:
        model_training_yaml = components.create_component_from_func(
            func=model_training,
            base_image='python:3.9',
            packages_to_install=[
                'pandas>=2.0.0',
                'numpy>=1.24.0',
                'scikit-learn>=1.3.0',
                'joblib>=1.3.0'
            ]
        )
        
        output_path = components_dir / 'model_training.yaml'
        with open(output_path, 'w') as f:
            f.write(model_training_yaml)
        
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Component 4: Model Evaluation
    print("\n4. Compiling Model Evaluation component...")
    try:
        model_evaluation_yaml = components.create_component_from_func(
            func=model_evaluation,
            base_image='python:3.9',
            packages_to_install=[
                'pandas>=2.0.0',
                'numpy>=1.24.0',
                'scikit-learn>=1.3.0',
                'joblib>=1.3.0'
            ]
        )
        
        output_path = components_dir / 'model_evaluation.yaml'
        with open(output_path, 'w') as f:
            f.write(model_evaluation_yaml)
        
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("COMPILATION COMPLETE!")
    print("=" * 60)
    
    # List all compiled components
    print("\nCompiled components:")
    yaml_files = list(components_dir.glob('*.yaml'))
    if yaml_files:
        for yaml_file in sorted(yaml_files):
            file_size = yaml_file.stat().st_size
            print(f"  - {yaml_file.name} ({file_size} bytes)")
    else:
        print("  (No YAML files found - compilation may have failed)")
    
    print(f"\nComponents directory: {components_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    compile_all_components()
