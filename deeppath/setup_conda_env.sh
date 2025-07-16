#!/bin/bash
# Script to create and setup a conda environment for the DeepPath PyTorch implementation

# Set environment name
ENV_NAME="deeppath_torch"

# Check for conda
command -v conda >/dev/null 2>&1 || { echo "Error: conda is required but not installed"; exit 1; }

# Check if environment already exists
if conda env list | grep -q "$ENV_NAME"; then
  read -p "Environment $ENV_NAME already exists. Do you want to recreate it? (y/n): " choice
  case "$choice" in 
    y|Y ) echo "Removing existing environment..." && conda env remove -n $ENV_NAME -y ;;
    * ) echo "Using existing environment..." ;;
  esac
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "$ENV_NAME"; then
  echo "Creating conda environment: $ENV_NAME"
  conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment
echo "Activating conda environment"
eval "$(conda shell.bash hook)" || { echo "Error: Failed to initialize conda shell"; exit 1; }
if ! conda activate $ENV_NAME; then
  echo "Error: Failed to activate $ENV_NAME environment"
  exit 1
fi

# Check if PyTorch is already installed
if ! python -c "import torch; print(f'PyTorch {torch.__version__} detected')" &>/dev/null; then
  # Install PyTorch with MPS support (for Apple Silicon)
  echo "Installing PyTorch with MPS support for Apple Silicon"
  conda install pytorch torchvision torchaudio -c pytorch -y
else
  echo "PyTorch already installed, skipping installation"
fi

# Check if other dependencies are installed
if ! python -c "import numpy, scipy, sklearn" &>/dev/null; then
  # Install other dependencies
  echo "Installing other dependencies"
  conda install numpy scipy scikit-learn -y
else
  echo "Dependencies already installed"
fi

# Check PyTorch installation with MPS support
echo "Checking PyTorch installation with MPS support"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

echo "Conda environment setup completed!"
echo "To activate the environment, run: conda activate $ENV_NAME"
echo "To test the PyTorch implementation, run: python test_torch.py"
echo "To run the full pipeline, run: ./pathfinder_torch.sh <relation>"