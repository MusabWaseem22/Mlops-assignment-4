# MLOps Kubeflow Assignment

A complete MLOps pipeline implementation for training and evaluating machine learning models using Kubeflow Pipelines, DVC for data versioning, and CI/CD automation.

## Project Overview

This project implements an end-to-end MLOps pipeline for the Boston Housing dataset prediction task. The pipeline orchestrates data extraction from versioned storage, data preprocessing, model training, and model evaluation using Kubeflow Pipelines deployed on Minikube.

### ML Problem

**Task:** Regression problem to predict median house prices in the Boston area  
**Dataset:** Boston Housing Dataset (506 samples, 13 features)  
**Model:** Random Forest Regressor  
**Metrics:** R2 Score, RMSE, MSE, MAE

### Pipeline Components

The pipeline consists of four main components:

1. **Data Extraction** - Retrieves versioned dataset from DVC remote storage
2. **Data Preprocessing** - Cleans, scales, and splits data into train/test sets
3. **Model Training** - Trains a Random Forest model on the training data
4. **Model Evaluation** - Evaluates model performance and saves metrics

## Project Structure

```
mlops-kubeflow-assignment/
├── data/
│   ├── raw/                      # Raw data files (tracked by DVC)
│   │   ├── raw_data.csv          # Boston Housing dataset
│   │   └── raw_data.csv.dvc      # DVC metadata file
│   └── processed/                # Processed data files
│       ├── train.csv             # Training dataset
│       └── test.csv              # Test dataset
├── src/
│   ├── pipeline_components.py    # Kubeflow component definitions
│   └── model_training.py         # Standalone training script
├── components/                   # Compiled Kubeflow pipeline YAML files
│   ├── data_extraction.yaml
│   ├── data_preprocessing.yaml
│   ├── model_training.yaml
│   └── model_evaluation.yaml
├── scripts/
│   ├── create_dataset.py         # Dataset creation script
│   ├── compile_components.py     # Component compilation script
│   ├── compile_pipeline.py       # Pipeline compilation script
│   └── run_pipeline_local.py     # Local pipeline execution
├── models/                       # Trained model artifacts
├── metrics/                      # Evaluation metrics
├── pipeline.py                   # Main Kubeflow pipeline definition
├── pipeline.yaml                 # Compiled pipeline (generated)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Custom pipeline component image
├── Jenkinsfile                   # Jenkins CI/CD pipeline
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions workflow
├── .dvc/                         # DVC configuration
│   ├── config                    # DVC remote storage config
│   └── .gitignore
└── README.md                     # This file
```

## Setup Instructions

### Prerequisites

- **Python 3.9+**
- **Git**
- **Docker Desktop** (for Minikube driver)
- **Minikube** (for local Kubernetes cluster)
- **kubectl** (Kubernetes command-line tool)
- **DVC** (Data Version Control)

### 1. Install Minikube

**macOS (Apple Silicon):**
```bash
# Download Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-darwin-arm64
chmod +x minikube-darwin-arm64
sudo mv minikube-darwin-arm64 /usr/local/bin/minikube

# Or using Homebrew
brew install minikube
```

**Linux:**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

**Verify Installation:**
```bash
minikube version
```

### 2. Start Minikube

```bash
# Start Minikube with sufficient resources
minikube start --driver=docker --memory=8192 --cpus=4 --disk-size=50g

# Verify Minikube is running
minikube status
```

**Expected Output:**
```
minikube
type: Control Plane
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured
```

### 3. Deploy Kubeflow Pipelines

```bash
# Create kubeflow namespace
kubectl create namespace kubeflow

# Deploy Kubeflow Pipelines (standalone)
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.0.5"

# Wait for pods to be ready (5-10 minutes)
kubectl get pods -n kubeflow -w

# Access KFP UI via port-forward
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

**Access UI:** Open http://localhost:8080 in your browser

### 4. Setup DVC Remote Storage

```bash
# Navigate to project directory
cd mlops-kubeflow-assignment

# Initialize DVC (if not already done)
dvc init

# Configure remote storage (local directory)
dvc remote add -d storage ../dvc-storage

# Alternative: Use Google Drive
# dvc remote add -d storage gdrive://your-folder-id

# Alternative: Use AWS S3
# dvc remote add -d storage s3://your-bucket/dvc-cache

# Track data file
dvc add data/raw/raw_data.csv

# Push to remote storage
dvc push

# Commit DVC files to Git
git add data/raw/raw_data.csv.dvc .dvc/config
git commit -m "Add data tracking with DVC"
```

### 5. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Prepare Dataset

```bash
# Generate Boston Housing dataset
python3 scripts/create_dataset.py
```

## Pipeline Walkthrough

### Compiling the Pipeline

The pipeline must be compiled from Python code to a YAML file that can be uploaded to Kubeflow Pipelines.

```bash
# Compile pipeline to YAML
python3 pipeline.py
```

This creates `pipeline.yaml` in the project root directory.

**What happens during compilation:**
- Validates pipeline structure
- Resolves component dependencies
- Generates pipeline DAG (Directed Acyclic Graph)
- Creates portable YAML specification

### Pipeline Components

The pipeline consists of four reusable components:

#### 1. Data Extraction
- **Inputs:** DVC remote URL, data path
- **Outputs:** Extracted data file path
- **Function:** Fetches versioned dataset from DVC remote storage

#### 2. Data Preprocessing
- **Inputs:** Raw data path, preprocessing parameters
- **Outputs:** Training and test dataset paths
- **Function:** Cleans, scales features, and splits data

#### 3. Model Training
- **Inputs:** Training data path, model hyperparameters
- **Outputs:** Trained model artifact path
- **Function:** Trains Random Forest model and saves it

#### 4. Model Evaluation
- **Inputs:** Model path, test data path, evaluation threshold
- **Outputs:** Evaluation metrics file
- **Function:** Evaluates model and saves metrics (R2, RMSE, etc.)

### Running the Pipeline

#### Option 1: Using Kubeflow Pipelines UI

1. **Access KFP UI:** http://localhost:8080

2. **Upload Pipeline:**
   - Click "Upload Pipeline"
   - Select `pipeline.yaml`
   - Name it: "Boston Housing ML Pipeline"

3. **Create Run:**
   - Click on uploaded pipeline
   - Click "Create Run"
   - Configure parameters:
     - `dvc_remote_url`: `../../dvc-storage`
     - `dvc_data_path`: `data/raw/raw_data.csv`
     - `test_size`: `0.2`
     - `random_state`: `42`
     - `n_estimators`: `100`
     - `threshold_r2`: `0.5`
   - Click "Start"

4. **Monitor Execution:**
   - Watch pipeline graph in UI
   - Check component logs
   - View outputs and metrics

#### Option 2: Using KFP SDK

```python
import kfp
from pipeline import boston_housing_pipeline

# Connect to KFP
client = kfp.Client(host='http://localhost:8080')

# Upload pipeline
client.upload_pipeline(
    pipeline_package_path='pipeline.yaml',
    pipeline_name='boston-housing-ml-pipeline'
)

# Create and run
run_result = client.create_run_from_pipeline_package(
    pipeline_file='pipeline.yaml',
    arguments={
        'dvc_remote_url': '../../dvc-storage',
        'dvc_data_path': 'data/raw/raw_data.csv',
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100,
        'threshold_r2': 0.5
    }
)
```

#### Option 3: Run Components Locally

```bash
# Run all components locally (for testing)
python3 scripts/run_pipeline_local.py

# View metrics
cat metrics/evaluation_metrics.txt
```

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dvc_remote_url` | str | `../../dvc-storage` | Path to DVC remote storage |
| `dvc_data_path` | str | `data/raw/raw_data.csv` | Path to data file in DVC |
| `test_size` | float | `0.2` | Proportion of data for testing |
| `random_state` | int | `42` | Random seed for reproducibility |
| `n_estimators` | int | `100` | Number of trees in Random Forest |
| `max_depth` | int | `0` | Max tree depth (0 = unlimited) |
| `min_samples_split` | int | `2` | Min samples to split a node |
| `threshold_r2` | float | `0.5` | Minimum R2 score threshold |
| `scale_features` | bool | `True` | Whether to scale features |

## CI/CD Pipeline

### Jenkins Pipeline

The `Jenkinsfile` defines a declarative pipeline with three stages:

**Stage 1: Environment Setup**
- Checks out code from GitHub
- Installs Python 3.9
- Creates virtual environment
- Installs dependencies from `requirements.txt`

**Stage 2: Pipeline Compilation**
- Compiles `pipeline.py` to `pipeline.yaml`
- Validates YAML syntax
- Verifies pipeline file was created

**Stage 3: Validation**
- Validates component YAML files exist
- Checks pipeline structure
- Verifies component dependencies

**Setup Jenkins Job:**
1. Install Jenkins
2. Create new Pipeline job
3. Configure to use Jenkinsfile from SCM (GitHub)
4. Point to this repository's main branch
5. Run job manually or configure webhook

### GitHub Actions

The `.github/workflows/ci.yml` file defines a GitHub Actions workflow that:
- Runs on push/PR to main/master
- Executes the same three stages as Jenkins
- Validates pipeline compilation
- Uploads compiled pipeline as artifact

**View Workflow:**
- Go to repository → Actions tab
- See workflow runs and status

## Data Versioning with DVC

### DVC Configuration

DVC is configured to track the Boston Housing dataset:

```bash
# Check DVC status
dvc status

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

### Remote Storage Options

- **Local Directory:** `../dvc-storage` (default)
- **Google Drive:** `gdrive://folder-id`
- **AWS S3:** `s3://bucket-name/path`
- **Azure Blob:** `azure://container-name`

## Usage Examples

### Compile Pipeline

```bash
python3 pipeline.py
```

### Run Components Locally

```bash
# Run individual component
python3 -c "from src.pipeline_components import data_extraction; data_extraction('../../dvc-storage', 'data/raw/raw_data.csv', 'output.csv')"

# Run complete pipeline locally
python3 scripts/run_pipeline_local.py
```

### View Pipeline Structure

```bash
# Generate pipeline graph
python3 scripts/generate_pipeline_graph.py
```

## Troubleshooting

### Minikube Issues

**Problem:** Minikube won't start
```bash
minikube delete
minikube start --driver=docker --memory=8192 --cpus=4
```

**Problem:** Out of memory
```bash
# Reduce memory allocation
minikube start --memory=4096 --cpus=2
```

### Kubeflow Pipelines Issues

**Problem:** Pods not starting
```bash
# Check pod status
kubectl get pods -n kubeflow

# Check pod logs
kubectl logs -n kubeflow <pod-name>

# Restart deployment
kubectl rollout restart deployment/<deployment-name> -n kubeflow
```

**Problem:** Image pull errors
- Some images may not be available for ARM64 architecture
- Consider using AMD64-based Minikube or alternative image sources

### Pipeline Compilation Errors

**Problem:** Import errors
```bash
# Ensure kfp is installed
pip install kfp>=2.4.0

# Verify Python path
python3 -c "import kfp; print(kfp.__version__)"
```

**Problem:** Component loading errors
```bash
# Verify component YAML files exist
ls -lh components/*.yaml

# Check component structure
cat components/data_extraction.yaml | head -20
```

### DVC Issues

**Problem:** DVC push fails
```bash
# Check remote configuration
dvc remote list

# Verify remote path exists
ls -la ../dvc-storage

# Reconfigure remote
dvc remote remove storage
dvc remote add -d storage ../dvc-storage
```

## Project Deliverables

### Task 1: Project Initialization and Data Versioning
- ✅ Project structure created
- ✅ DVC initialized and configured
- ✅ Dataset tracked with DVC

### Task 2: Building Kubeflow Pipeline Components
- ✅ Four pipeline components implemented
- ✅ Components compiled to YAML files
- ✅ Component inputs/outputs documented

### Task 3: Orchestrating the Pipeline on Minikube
- ✅ Pipeline orchestration defined
- ✅ Pipeline compiled to YAML
- ✅ Ready for execution on Kubeflow

### Task 4: Continuous Integration
- ✅ Jenkinsfile with 3 stages
- ✅ GitHub Actions workflow configured
- ✅ Pipeline compilation automated

### Task 5: Documentation
- ✅ Comprehensive README.md created
- ✅ Setup instructions documented
- ✅ Pipeline walkthrough included

## File Descriptions

- **`pipeline.py`** - Main pipeline definition orchestrating all components
- **`src/pipeline_components.py`** - Python functions for each pipeline component
- **`components/*.yaml`** - Compiled Kubeflow component specifications
- **`pipeline.yaml`** - Compiled pipeline ready for upload to KFP
- **`Jenkinsfile`** - Jenkins declarative pipeline configuration
- **`.github/workflows/ci.yml`** - GitHub Actions CI/CD workflow
- **`requirements.txt`** - Python package dependencies

## Contributing

This is an assignment project. For questions or issues, please refer to the assignment documentation.

## License

MIT License

## References

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [DVC Documentation](https://dvc.org/doc)
- [Minikube Documentation](https://minikube.sigs.k8s.io/docs/)
