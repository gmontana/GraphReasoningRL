# MINERVA Implementation Comparison Guide

This directory contains tools to verify parity between the original TensorFlow MINERVA implementation and our PyTorch port.

## Quick Start

### 1. Setup Environments

Create separate virtual environments for TensorFlow and PyTorch:

```bash
# TensorFlow environment
python -m venv venv_tf
source venv_tf/bin/activate  # On Windows: venv_tf\Scripts\activate
pip install -r requirements_tf.txt

# PyTorch environment  
python -m venv venv_pytorch
source venv_pytorch/bin/activate  # On Windows: venv_pytorch\Scripts\activate
pip install -r requirements_pytorch.txt
```

### 2. Run Comparison

Compare both implementations on a dataset:

```bash
python compare_implementations.py --dataset countries_S1 --iterations 100
```

Test multiple datasets:

```bash
python compare_implementations.py --test_multiple --iterations 100
```

### 3. Run Individual Implementations

Run TensorFlow only:
```bash
python run_tensorflow.py --dataset kinship --iterations 200
```

Run PyTorch only:
```bash
python run_pytorch.py --dataset kinship --iterations 200
```

## Files in this Directory

- `compare_implementations.py` - Main comparison script that runs both implementations
- `run_tensorflow.py` - Runs the original TensorFlow MINERVA
- `run_pytorch.py` - Runs our PyTorch MINERVA
- `parity_tests.py` - Unit tests for critical functions
- `critical_functions_comparison.py` - Side-by-side code comparison
- `requirements_tf.txt` - TensorFlow dependencies
- `requirements_pytorch.txt` - PyTorch dependencies

## Verification Results

### Critical Bug Fixed
During comparison, we found and fixed a bug in the cumulative discounted reward calculation:

```python
# INCORRECT (original PyTorch):
for t in reversed(range(self.path_length - 1)):
    cum_disc_reward[:, t] = cum_disc_reward[:, t + 1] * self.gamma

# CORRECT (fixed):
for t in reversed(range(self.path_length)):
    running_add = self.gamma * running_add + cum_disc_reward[:, t]
    cum_disc_reward[:, t] = running_add
```

### Parity Test Results
All critical functions pass parity tests:
- ✅ Cumulative Reward Calculation
- ✅ Entropy Regularization
- ✅ Action Masking and Scoring  
- ✅ Beam Search Indexing

### Expected Results

Due to framework differences in random initialization and sampling, exact numerical matches are not expected. However, the implementations should produce results within 5% relative difference when:

1. Using same hyperparameters
2. Running for sufficient iterations (>100)
3. Evaluating on same test data

### Datasets Available

Small datasets good for testing (fast training):
- `countries_S1`, `countries_S2`, `countries_S3` - Synthetic country relationships
- `kinship` - Family relationship reasoning

Larger datasets (slower training):
- `WN18RR` - WordNet subset
- `FB15K-237` - Freebase subset (if available)

## Troubleshooting

### TensorFlow Issues
- Requires TensorFlow 1.15 which needs Python 3.7
- May show deprecation warnings (normal)
- Use `TF_CPP_MIN_LOG_LEVEL=2` to suppress warnings

### PyTorch Issues  
- Ensure CUDA version matches PyTorch if using GPU
- CPU training is much slower but works for testing

### Comparison Differences
If results differ significantly:
1. Increase training iterations
2. Try different random seeds
3. Check for numerical overflow in small datasets
4. Verify same preprocessing is applied

## Implementation Details Verified

1. **Agent Architecture**: LSTM + MLP policy network ✓
2. **Environment**: Episode generation and rewards ✓
3. **Training**: REINFORCE with baseline ✓
4. **Evaluation**: Beam search with deduplication ✓
5. **Data Loading**: Graph structure and batching ✓

The PyTorch implementation maintains functional parity with TensorFlow while offering cleaner code and better performance.