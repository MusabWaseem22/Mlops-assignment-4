# MLOps Kubeflow Assignment

This repository contains a complete MLOps pipeline implementation using Kubeflow Pipelines, DVC for data versioning, and CI/CD integration.

## Project Structure

```
mlops-kubeflow-assignment/
├── data/
│   ├── raw/          # Raw data files (tracked by DVC)
│   └── processed/    # Processed data files
├── src/
│   ├── pipeline_components.py  # Kubeflow component definitions
│   └── model_training.py       # Training script
├── components/       # Compiled Kubeflow pipeline YAML files
├── pipeline.py       # Main Kubeflow pipeline definition
├── requirements.txt  # Python dependencies
├── Dockerfile        # Custom pipeline component image
├── Jenkinsfile       # Jenkins CI/CD pipeline
├── .github/workflows/ci.yml  # GitHub Actions workflow
└── .dvc/            # DVC configuration

```

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Git
- DVC
- Docker (optional, for custom components)
- Kubeflow Pipelines (on Minikube)
- kubectl configured for Minikube

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mlops-kubeflow-assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. DVC Setup

```bash
# Initialize DVC (if not already done)
dvc init

# Configure remote storage (example: local directory)
dvc remote add -d storage /path/to/dvc/storage

# Or use Google Drive, S3, etc.
# dvc remote add -d storage gdrive://your-folder-id
# dvc remote add -d storage s3://your-bucket/dvc-cache

# Track data
dvc add data/raw/raw_data.csv

# Commit to git
git add data/raw/raw_data.csv.dvc .dvc/config
git commit -m "Add data tracking with DVC"
```

### 4. Dataset

The project uses the Boston Housing dataset. Run the data preparation script to download and prepare the data:

```bash
python scripts/prepare_data.py
```

## Usage

### Running the Pipeline Locally

```bash
# Compile the pipeline
python pipeline.py

# The compiled YAML will be in components/boston_housing_pipeline.yaml
```

### Running on Kubeflow

1. Upload the compiled pipeline YAML to Kubeflow Pipelines UI
2. Or use the KFP SDK to run programmatically

## CI/CD

### GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Runs tests on push/PR
- Checks code quality with flake8
- Verifies DVC setup
- Builds Docker image

### Jenkins

The `Jenkinsfile` defines a declarative pipeline for Jenkins that:
- Checks out code
- Sets up Python environment
- Runs linting and tests
- Checks DVC status
- Builds Docker image

## License

MIT

