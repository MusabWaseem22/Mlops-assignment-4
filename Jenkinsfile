pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
    }
    
    stages {
        // Stage 1: Environment Setup
        stage('Environment Setup') {
            steps {
                echo 'Stage 1: Environment Setup'
                echo '========================='
                
                // Checkout code from GitHub
                checkout scm
                
                echo 'Code checked out successfully'
                
                // Install Python dependencies
                sh '''
                    echo "Setting up Python environment..."
                    python${PYTHON_VERSION} --version
                    
                    # Create virtual environment
                    python${PYTHON_VERSION} -m venv venv || python3 -m venv venv
                    source venv/bin/activate || . venv/bin/activate
                    
                    # Upgrade pip
                    pip install --upgrade pip
                    
                    # Install dependencies from requirements.txt
                    echo "Installing dependencies from requirements.txt..."
                    pip install -r requirements.txt
                    
                    echo "Environment setup complete!"
                    pip list
                '''
            }
        }
        
        // Stage 2: Pipeline Compilation
        stage('Pipeline Compilation') {
            steps {
                echo 'Stage 2: Pipeline Compilation'
                echo '============================='
                
                sh '''
                    source venv/bin/activate || . venv/bin/activate
                    
                    echo "Compiling Kubeflow pipeline..."
                    echo "Running: python3 pipeline.py"
                    
                    # Compile pipeline to ensure it is syntactically correct
                    python3 pipeline.py
                    
                    # Verify pipeline.yaml was created
                    if [ -f "pipeline.yaml" ]; then
                        echo "SUCCESS: pipeline.yaml generated successfully!"
                        ls -lh pipeline.yaml
                        echo ""
                        echo "Pipeline file size:"
                        wc -l pipeline.yaml
                    else
                        echo "ERROR: pipeline.yaml not found!"
                        exit 1
                    fi
                    
                    # Verify pipeline.yaml is valid YAML
                    echo ""
                    echo "Validating pipeline.yaml syntax..."
                    python3 -c "import yaml; yaml.safe_load(open('pipeline.yaml'))" && echo "YAML syntax is valid!"
                    
                    echo "Pipeline compilation completed successfully!"
                '''
            }
        }
        
        // Stage 3: Validation
        stage('Validation') {
            steps {
                echo 'Stage 3: Validation'
                echo '==================='
                
                sh '''
                    source venv/bin/activate || . venv/bin/activate
                    
                    echo "Validating pipeline components..."
                    
                    # Check that all component YAML files exist
                    echo "Checking component files..."
                    if [ -f "components/data_extraction.yaml" ] && \
                       [ -f "components/data_preprocessing.yaml" ] && \
                       [ -f "components/model_training.yaml" ] && \
                       [ -f "components/model_evaluation.yaml" ]; then
                        echo "All component YAML files present"
                        ls -lh components/*.yaml
                    else
                        echo "WARNING: Some component files missing"
                    fi
                    
                    # Validate pipeline structure
                    echo ""
                    echo "Validating pipeline structure..."
                    python3 -c "
from kfp import compiler
import yaml
with open('pipeline.yaml', 'r') as f:
    spec = yaml.safe_load(f)
    tasks = spec.get('root', {}).get('dag', {}).get('tasks', {})
    print(f'Pipeline contains {len(tasks)} components')
    for task in tasks:
        print(f'  - {task}')
"
                    
                    echo ""
                    echo "Validation completed successfully!"
                '''
            }
        }
    }
    
    post {
        always {
            echo 'Pipeline execution completed'
            // Optional: Archive pipeline.yaml as artifact
            archiveArtifacts artifacts: 'pipeline.yaml', allowEmptyArchive: false
        }
        success {
            echo 'All stages completed successfully!'
            echo 'Jenkins pipeline passed!'
        }
        failure {
            echo 'Pipeline failed. Check logs above for details.'
        }
    }
}
