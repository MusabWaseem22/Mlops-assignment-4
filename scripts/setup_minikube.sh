#!/bin/bash
# Setup script for Minikube and Kubeflow Pipelines
# This script automates the setup process for Task 3

set -e

echo "============================================================"
echo "Minikube and Kubeflow Pipelines Setup"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo -e "${RED}ERROR: minikube is not installed${NC}"
    echo "Please install minikube first:"
    echo "  macOS: brew install minikube"
    echo "  Linux: See https://minikube.sigs.k8s.io/docs/start/"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}ERROR: kubectl is not installed${NC}"
    echo "Please install kubectl first"
    exit 1
fi

# Check if docker is running
if ! docker info &> /dev/null; then
    echo -e "${YELLOW}WARNING: Docker is not running${NC}"
    echo "Starting Docker..."
    # Try to start Docker (platform-specific)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Docker
        echo "Waiting for Docker to start..."
        sleep 10
    fi
fi

echo -e "${GREEN}Step 1: Starting Minikube...${NC}"
minikube start \
    --driver=docker \
    --memory=8192 \
    --cpus=4 \
    --disk-size=50g

echo -e "${GREEN}Step 2: Verifying Minikube status...${NC}"
minikube status

echo -e "${GREEN}Step 3: Creating kubeflow namespace...${NC}"
kubectl create namespace kubeflow --dry-run=client -o yaml | kubectl apply -f -

echo -e "${GREEN}Step 4: Deploying Kubeflow Pipelines...${NC}"
echo "This may take several minutes..."

# Deploy Kubeflow Pipelines using kustomize
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.0.5"

echo -e "${GREEN}Step 5: Waiting for pods to be ready...${NC}"
echo "This may take 5-10 minutes..."

kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=600s || true

echo -e "${GREEN}Step 6: Setting up port forwarding...${NC}"
echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "To access Kubeflow Pipelines UI, run:"
echo "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
echo ""
echo "Then open: http://localhost:8080"
echo ""
echo "To check Minikube status:"
echo "  minikube status"
echo ""
echo "To check pipeline pods:"
echo "  kubectl get pods -n kubeflow"
echo ""

