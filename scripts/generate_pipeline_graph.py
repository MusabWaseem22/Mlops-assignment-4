#!/usr/bin/env python3
"""
Generate pipeline graph visualization from pipeline.yaml
This creates a visual representation of the pipeline structure
"""

import yaml
import json

def generate_pipeline_graph():
    """Read pipeline.yaml and generate graph structure"""
    
    print("=" * 60)
    print("PIPELINE GRAPH STRUCTURE")
    print("=" * 60)
    
    try:
        with open('pipeline.yaml', 'r') as f:
            pipeline_spec = yaml.safe_load(f)
        
        print("\nPipeline Name:", pipeline_spec.get('pipelineInfo', {}).get('name', 'N/A'))
        print("\nPipeline Components:")
        print("-" * 60)
        
        # Extract component information
        if 'root' in pipeline_spec and 'dag' in pipeline_spec['root']:
            tasks = pipeline_spec['root']['dag'].get('tasks', {})
            
            print("\nComponents and Dependencies:")
            for task_name, task_info in tasks.items():
                component_ref = task_info.get('componentRef', {}).get('name', task_name)
                print(f"\n{task_name}:")
                print(f"  Component: {component_ref}")
                
                # Get dependencies
                dependencies = task_info.get('dependentTasks', [])
                if dependencies:
                    print(f"  Depends on: {', '.join(dependencies)}")
                else:
                    print(f"  Depends on: None (start of pipeline)")
        
        print("\n" + "=" * 60)
        print("Pipeline Flow:")
        print("=" * 60)
        print("1. Extract Data from DVC")
        print("   ↓")
        print("2. Preprocess Data")
        print("   ↓")
        print("3. Train Random Forest Model")
        print("   ↓")
        print("4. Evaluate Model")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error reading pipeline: {e}")

if __name__ == '__main__':
    generate_pipeline_graph()

