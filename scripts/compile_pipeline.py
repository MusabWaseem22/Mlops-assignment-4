#!/usr/bin/env python3
"""
Script to compile the Kubeflow pipeline to YAML format.

This script compiles the pipeline definition from pipeline.py into
a YAML file that can be uploaded to Kubeflow Pipelines.

Usage:
    python3 scripts/compile_pipeline.py
    
Output:
    pipeline.yaml - Compiled pipeline ready for upload to KFP
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from kfp import compiler
    from pipeline import boston_housing_pipeline
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please ensure kfp is installed:")
    print("  pip install kfp>=2.4.0")
    sys.exit(1)


def compile_pipeline():
    """Compile the pipeline to YAML format."""
    
    pipeline_dir = Path(__file__).parent.parent
    output_file = pipeline_dir / 'pipeline.yaml'
    
    print("=" * 60)
    print("COMPILING KUBEFLOW PIPELINE")
    print("=" * 60)
    print(f"\nPipeline function: boston_housing_pipeline")
    print(f"Output file: {output_file}")
    print("\nCompiling...")
    
    try:
        compiler.Compiler().compile(
            pipeline_func=boston_housing_pipeline,
            package_path=str(output_file)
        )
        
        # Verify file was created
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"\n[SUCCESS] Pipeline compiled successfully!")
            print(f"Output: {output_file}")
            print(f"Size: {file_size} bytes")
            print("\n" + "=" * 60)
            print("NEXT STEPS:")
            print("=" * 60)
            print("1. Start Minikube: minikube start")
            print("2. Deploy Kubeflow Pipelines (see TASK3_SETUP.md)")
            print("3. Upload pipeline.yaml to Kubeflow Pipelines UI")
            print("4. Run the pipeline with appropriate parameters")
            print("=" * 60 + "\n")
        else:
            print("\n[ERROR] Pipeline file was not created!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Failed to compile pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    compile_pipeline()

