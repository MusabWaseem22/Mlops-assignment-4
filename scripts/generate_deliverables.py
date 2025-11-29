#!/usr/bin/env python3
"""
Helper script to generate deliverables for Task 1.
This script helps create the necessary outputs for submission.
"""

import os
import subprocess
import sys
from pathlib import Path


def print_file_structure():
    """Print the project file structure."""
    print("=" * 60)
    print("PROJECT FILE STRUCTURE")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent
    
    # List of files to show
    important_files = [
        "Dockerfile",
        "Jenkinsfile",
        "pipeline.py",
        "requirements.txt",
        "README.md",
        "SETUP_SUMMARY.md",
        ".gitignore",
        ".dvc/.gitignore",
        ".dvc/config",
        "src/pipeline_components.py",
        "src/model_training.py",
        "data/raw/raw_data.csv.dvc",
        ".github/workflows/ci.yml",
    ]
    
    print("\nProject Structure:\n")
    print("mlops-kubeflow-assignment/")
    
    for file_path in sorted(base_path.rglob("*")):
        if file_path.is_file() and file_path.name not in ['.DS_Store', '__pycache__']:
            rel_path = file_path.relative_to(base_path)
            depth = len(rel_path.parts) - 1
            
            # Skip .git directory
            if '.git' in rel_path.parts:
                continue
                
            indent = "  " * depth
            print(f"{indent}├── {rel_path.name}")
    
    print("\n" + "=" * 60)


def show_dvc_status():
    """Run and show DVC status."""
    print("\n" + "=" * 60)
    print("DVC STATUS")
    print("=" * 60 + "\n")
    
    base_path = Path(__file__).parent.parent
    os.chdir(base_path)
    
    try:
        result = subprocess.run(
            ["python3", "-m", "dvc", "status"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running dvc status: {e}")
        print(e.stderr)


def show_dvc_push():
    """Run and show DVC push."""
    print("\n" + "=" * 60)
    print("DVC PUSH")
    print("=" * 60 + "\n")
    
    base_path = Path(__file__).parent.parent
    os.chdir(base_path)
    
    try:
        result = subprocess.run(
            ["python3", "-m", "dvc", "push"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running dvc push: {e}")
        print(e.stderr)


def show_requirements():
    """Show requirements.txt content."""
    print("\n" + "=" * 60)
    print("REQUIREMENTS.TXT CONTENT")
    print("=" * 60 + "\n")
    
    base_path = Path(__file__).parent.parent
    requirements_path = base_path / "requirements.txt"
    
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            print(f.read())
    else:
        print("requirements.txt not found!")


def show_dvc_config():
    """Show DVC configuration."""
    print("\n" + "=" * 60)
    print("DVC CONFIGURATION")
    print("=" * 60 + "\n")
    
    base_path = Path(__file__).parent.parent
    dvc_config_path = base_path / ".dvc" / "config"
    
    if dvc_config_path.exists():
        with open(dvc_config_path, 'r') as f:
            print(f.read())
    else:
        print(".dvc/config not found!")


def main():
    """Main function to generate all deliverables."""
    print("\n" + "=" * 60)
    print("TASK 1 DELIVERABLES GENERATOR")
    print("=" * 60)
    
    # Show file structure
    print_file_structure()
    
    # Show DVC status
    show_dvc_status()
    
    # Show DVC push
    show_dvc_push()
    
    # Show requirements.txt
    show_requirements()
    
    # Show DVC config
    show_dvc_config()
    
    print("\n" + "=" * 60)
    print("DELIVERABLES READY FOR SCREENSHOTS")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Take screenshots of the outputs above")
    print("2. Create GitHub repository: mlops-kubeflow-assignment")
    print("3. Push code to GitHub")
    print("4. Take screenshot of GitHub repository file structure")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

