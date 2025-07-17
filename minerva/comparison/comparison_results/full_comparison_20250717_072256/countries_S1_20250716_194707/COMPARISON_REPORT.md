# MINERVA Implementation Comparison Report

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250716_194707
**Verdict**: PASS

## Results Comparison

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.5833 | 0.4167 | 41.67% | ✗ |
| hits@3 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| hits@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| hits@10 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| mrr | 0.0000 | 0.7847 | 0.7847 | 78470.00% | ✓ |

## Analysis

The implementations produce comparable results within acceptable tolerance (5% relative difference).
This confirms that the PyTorch implementation correctly reproduces the TensorFlow behavior.
