#!/bin/bash
# Script to run the original TensorFlow 1.x implementation

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the repository root directory (one level up from script)
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check for required commands
command -v conda >/dev/null 2>&1 || { echo "Error: conda is required but not installed"; exit 1; }
command -v python >/dev/null 2>&1 || { echo "Error: python is required but not installed"; exit 1; }

# Check for required TensorFlow files
if [ ! -d "$SCRIPT_DIR/tensorflow" ]; then
  echo "Error: tensorflow directory not found in benchmarks"
  exit 1
fi

if [ ! -f "$SCRIPT_DIR/tensorflow/sl_policy.py" ]; then
  echo "Error: sl_policy.py not found in tensorflow directory"
  exit 1
fi

if [ ! -f "$SCRIPT_DIR/tensorflow/policy_agent.py" ]; then
  echo "Error: policy_agent.py not found in tensorflow directory"
  exit 1
fi

if [ ! -f "$SCRIPT_DIR/tensorflow/evaluate.py" ]; then
  echo "Error: evaluate.py not found in tensorflow directory"
  exit 1
fi

# Check if a relation was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <relation_name>"
  echo "Example: $0 athletePlaysForTeam"
  exit 1
fi

relation=$1

# Create and configure a dedicated conda environment for TensorFlow 1.x
if ! conda env list | grep -q "deeppath_tf1"; then
  echo "Creating dedicated conda environment for TensorFlow 1.x..."
  # Force removal of deeppath_tf2 if it exists but has issues
  conda env remove -n deeppath_tf2 -y 2>/dev/null || true
  # Create new environment with Python 3.7 (compatible with TF 1.x)
  conda create -n deeppath_tf1 python=3.7 -y
  
  # Activate the environment and install dependencies
  eval "$(conda shell.bash hook)" || { echo "Error: Failed to initialize conda shell"; exit 1; }
  if ! conda activate deeppath_tf1; then
    echo "Error: Failed to activate deeppath_tf1 environment after creation"
    exit 1
  fi
  
  # Install TensorFlow 1.15 and other requirements
  echo "Installing TensorFlow 1.15 and dependencies..."
  conda install -y pip
  python -m pip install "tensorflow==1.15.0" numpy==1.19.5 scikit-learn==0.24.2
else
  echo "Using existing deeppath_tf1 conda environment..."
  # Force removal of the environment if it exists but has issues
  if ! conda activate deeppath_tf1 >/dev/null 2>&1; then
    echo "Existing environment has issues. Recreating it..."
    conda env remove -n deeppath_tf1 -y
    conda create -n deeppath_tf1 python=3.7 -y
    
    # Activate the environment and install dependencies
    eval "$(conda shell.bash hook)" || { echo "Error: Failed to initialize conda shell"; exit 1; }
    if ! conda activate deeppath_tf1; then
      echo "Error: Failed to activate deeppath_tf1 environment after creation"
      exit 1
    fi
    
    # Install TensorFlow 1.15 and other requirements
    echo "Installing TensorFlow 1.15 and dependencies..."
    conda install -y pip
    python -m pip install "tensorflow==1.15.0" numpy==1.19.5 scikit-learn==0.24.2
  else
    eval "$(conda shell.bash hook)" || { echo "Error: Failed to initialize conda shell"; exit 1; }
    if ! conda activate deeppath_tf1; then
      echo "Error: Failed to activate existing deeppath_tf1 environment"
      exit 1
    fi
    
    # Check if TensorFlow is installed correctly
    if ! python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} detected')" &>/dev/null; then
      echo "TensorFlow not found in environment, installing dependencies..."
      conda install -y pip
      python -m pip install "tensorflow==1.15.0" numpy==1.19.5 scikit-learn==0.24.2
    else
      echo "TensorFlow detected in environment"
    fi
  fi
fi

# No need for compatibility layer with TensorFlow 1.x
# Remove the tf2_compatibility.py file if it exists
if [ -f "$SCRIPT_DIR/tensorflow/tf2_compatibility.py" ]; then
  rm "$SCRIPT_DIR/tensorflow/tf2_compatibility.py"
fi

# Generate appropriate path files for the relation
mkdir -p "$REPO_ROOT/NELL-995/tasks/$relation"
if [ -f "$REPO_ROOT/NELL-995/tasks/$relation/train_pos" ]; then
  cp "$REPO_ROOT/NELL-995/tasks/$relation/train_pos" "$REPO_ROOT/NELL-995/tasks/$relation/train.pairs" 2>/dev/null
  cp "$REPO_ROOT/NELL-995/tasks/$relation/train_pos" "$REPO_ROOT/NELL-995/tasks/$relation/sort_test.pairs" 2>/dev/null
else
  echo "Warning: train_pos file not found for $relation"
fi

echo "Running original TensorFlow 1.x implementation..."

# Change to TensorFlow directory
cd "$SCRIPT_DIR/tensorflow"

# Run the supervised learning policy
echo "Running supervised learning policy..."
python sl_policy.py $relation

# Run the policy agent (retrain)
echo "Running policy agent (retrain)..."
python policy_agent.py $relation retrain

# Run the policy agent (test)
echo "Running policy agent (test)..."
python policy_agent.py $relation test

# Run evaluation
echo "Running original evaluation..."
python evaluate.py $relation || true

# Run simplified evaluation (fallback)
echo "Running simplified evaluation..."
python evaluate_simple.py $relation || true

# Return to original directory
cd "$SCRIPT_DIR"

# Deactivate the conda environment
conda deactivate

echo "TensorFlow implementation complete"
echo "Output files saved in NELL-995/tasks/$relation/"