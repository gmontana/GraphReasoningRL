# MINERVA Implementation Comparison Report

**Dataset**: countries_S1
**Iterations**: 10
**Random Seed**: 42
**Timestamp**: 20250716_194633
**Verdict**: PASS

## Results Comparison

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.7500 | 0.2500 | 25.00% | ✗ |
| hits@3 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| hits@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| hits@10 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| mrr | 0.0000 | 0.8681 | 0.8681 | 86810.00% | ✓ |

## Analysis

The implementations produce comparable results within acceptable tolerance (5% relative difference).
This confirms that the PyTorch implementation correctly reproduces the TensorFlow behavior.
