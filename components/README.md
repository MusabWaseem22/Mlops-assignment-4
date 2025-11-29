# Kubeflow Pipeline Components

This directory contains compiled YAML files for Kubeflow Pipeline components.

## Components

1. **data_extraction.yaml** - Extracts data from DVC remote storage
2. **data_preprocessing.yaml** - Preprocesses and splits data
3. **model_training.yaml** - Trains Random Forest model
4. **model_evaluation.yaml** - Evaluates model and saves metrics

## Compilation

To compile components from Python functions, run:

```bash
python3 scripts/compile_components.py
```

This requires `kfp>=2.4.0` to be installed.

## Usage in Pipeline

Components can be loaded in a pipeline using:

```python
from kfp import components

data_extraction_component = components.load_component_from_file('components/data_extraction.yaml')
```

