# MINERVA Implementation Comparison Report

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250716_192100
**Verdict**: FAIL

## Results Comparison

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0000 | 0.8333 | 0.8333 | 83330.00% | ✗ |
| hits@3 | 0.0000 | 1.0000 | 1.0000 | 100000.00% | ✗ |
| hits@5 | 0.0000 | 1.0000 | 1.0000 | 100000.00% | ✗ |
| hits@10 | 0.0000 | 1.0000 | 1.0000 | 100000.00% | ✗ |
| mrr | 0.0000 | 0.9097 | 0.9097 | 90970.00% | ✗ |

## Analysis

The implementations show significant differences in results.
Possible causes:
- Random initialization differences
- Numerical precision differences between frameworks
- Implementation bugs
- Different random sampling behavior

Recommendation: Run with more iterations or investigate specific differences.
