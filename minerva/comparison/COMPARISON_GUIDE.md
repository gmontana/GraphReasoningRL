# MINERVA Implementation Comparison Guide

This guide explains how to compare the TensorFlow and PyTorch implementations of MINERVA to verify parity and evaluate performance differences.

## Overview

The comparison framework allows you to:
- Run both implementations on the same dataset with identical hyperparameters
- Compare training validation metrics between implementations
- Generate detailed comparison reports
- Test multiple datasets in batch

**⚠️ Important**: The comparison evaluates **training validation metrics only**. The TensorFlow implementation has checkpoint compatibility issues that prevent test phase evaluation.

## Directory Structure

```
comparison/
├── COMPARISON_GUIDE.md      # This file
├── PARITY_REPORT.md         # Detailed parity analysis
├── compare.py               # Main comparison script
├── run_pytorch.py           # PyTorch runner
├── run_tensorflow.py        # TensorFlow runner
├── tf_patched/              # Modernized TensorFlow implementation
│   └── model/               # TF model components
├── outputs_pytorch/         # PyTorch results
├── outputs_tensorflow/      # TensorFlow results
└── comparison_results/      # Comparison reports
```

## Quick Start

### 1. Single Dataset Comparison

```bash
cd comparison/

# Compare on countries_S1 dataset with 100 iterations
python compare.py --dataset countries_S1 --iterations 100

# Compare on kinship dataset with 200 iterations
python compare.py --dataset kinship --iterations 200
```

### 2. Multiple Dataset Comparison

```bash
# Test on default set of datasets
python compare.py --test_multiple

# Test on specific datasets
python compare.py --datasets countries_S1 countries_S2 countries_S3 kinship umls
```

### 3. Individual Implementation Testing

```bash
# Run PyTorch implementation only
python run_pytorch.py --dataset countries_S1 --iterations 100

# Run TensorFlow implementation only
python run_tensorflow.py --dataset countries_S1 --iterations 100
```

## Understanding the Output

### Console Output

The comparison script provides real-time output showing:

1. **Implementation Status**:
   ```
   ✓ TensorFlow 2.15.0 available
   ✓ PyTorch 2.0.0 available
   ```

2. **Progress During Training**:
   - TensorFlow and PyTorch training logs are shown in real-time
   - You can see batch-by-batch progress

3. **Comparison Results**:
   ```
   Metric      TensorFlow   PyTorch      Difference   Status
   ------------------------------------------------------------
   hits@1      1.0000       0.9583       0.0417       ✓ Close
   hits@3      1.0000       1.0000       0.0000       ✓ Close
   hits@5      1.0000       1.0000       0.0000       ✓ Close
   hits@10     1.0000       1.0000       0.0000       ✓ Close
   mrr         1.0000       0.9792       0.0208       ✓ Close
   ```

### Generated Reports

1. **Individual Run Results**:
   - `outputs_pytorch/{dataset}/results.json`
   - `outputs_tensorflow/{dataset}/results.json`

2. **Comparison Report**:
   - `comparison_results/{dataset}_{timestamp}/COMPARISON_REPORT.md`
   - `comparison_results/{dataset}_{timestamp}/comparison_results.json`

3. **Multi-Dataset Summary**:
   - `comparison_results/MULTI_DATASET_SUMMARY_{timestamp}.md`

## Configuration Parameters

The comparison uses these default parameters:
- `iterations`: 100 (configurable)
- `batch_size`: 256
- `num_rollouts`: 20
- `test_rollouts`: 100
- `learning_rate`: 0.001
- `seed`: 42 (for reproducibility)

## Tolerance Settings

The comparison uses reasonable tolerances for RL algorithms:
- **Relative tolerance**: 10% (rtol=0.10)
- **Absolute tolerance**: 0.05 (atol=0.05)

These tolerances account for:
- Stochastic nature of RL training
- Framework-specific numerical differences
- Random initialization variations

## Interpreting Results

### Expected Differences

Small differences between implementations are normal due to:
1. **Framework differences**: PyTorch vs TensorFlow RNG implementations
2. **Numerical precision**: Floating-point computation differences
3. **Optimization**: Different optimizer implementations
4. **Initialization**: Minor variations in weight initialization

### What "Close" Means

- **✓ Close**: Metrics are within tolerance (difference < 10% relative or 0.05 absolute)
- **✗ Different**: Metrics exceed tolerance thresholds
- **- Missing**: One implementation doesn't report this metric

### TensorFlow Perfect Scores

The TensorFlow implementation often shows perfect validation scores (100%) on simple datasets like countries_S1. This is because:
- The validation is performed on the development set during training
- Simple datasets can be memorized during training
- The test phase (which would show more realistic scores) cannot be evaluated

## Troubleshooting

### Common Issues

1. **"TensorFlow not available"**:
   ```bash
   pip install tensorflow>=2.10.0
   ```

2. **"PyTorch not available"**:
   ```bash
   pip install torch>=1.10.0
   ```

3. **"Dataset not found"**:
   - Ensure datasets exist in `../original/datasets/data_preprocessed/`
   - Check dataset name spelling (case-sensitive)

4. **Out of Memory**:
   - Reduce batch_size in the dataset config
   - Use CPU instead of GPU for small experiments

### Debugging Tips

1. **Enable verbose output**:
   ```python
   # In compare.py, the output is already shown in real-time
   ```

2. **Check individual logs**:
   - TensorFlow logs: `outputs_tensorflow/{dataset}/train.log`
   - PyTorch logs: `outputs_pytorch/{dataset}/train.log`

3. **Verify dataset preprocessing**:
   - Check vocabulary files exist
   - Ensure graph.txt, train.txt, dev.txt are present

## Advanced Usage

### Custom Hyperparameters

Create a custom comparison with specific parameters:

```python
# Modify run_pytorch.py or run_tensorflow.py
config = {
    'batch_size': 512,
    'learning_rate': 0.0001,
    'beta': 0.02,
    'num_rollouts': 50
}
```

### Adding New Metrics

To track additional metrics in the comparison:

1. Update metric extraction in `run_pytorch.py` and `run_tensorflow.py`
2. Add metric to comparison loop in `compare.py`
3. Update tolerance settings if needed

## Summary

The comparison framework provides a robust way to verify that the PyTorch implementation maintains parity with the original TensorFlow version. While exact numerical matches are not expected due to framework differences, the implementations should produce similar performance metrics within reasonable tolerances.

For detailed parity analysis, see `PARITY_REPORT.md`.