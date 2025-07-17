# MINERVA Implementation Comparison Report

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250716_185758
**Verdict**: FAIL

## Results Comparison

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0000 | 0.6250 | 0.6250 | 62500.00% | ✗ |
| hits@3 | 0.0000 | 0.9583 | 0.9583 | 95830.00% | ✗ |
| hits@5 | 0.0000 | 1.0000 | 1.0000 | 100000.00% | ✗ |
| hits@10 | 0.0000 | 1.0000 | 1.0000 | 100000.00% | ✗ |
| mrr | 0.0000 | 0.7951 | 0.7951 | 79510.00% | ✗ |

## Analysis

The implementations show significant differences in results.
Possible causes:
- Random initialization differences
- Numerical precision differences between frameworks
- Implementation bugs
- Different random sampling behavior

Recommendation: Run with more iterations or investigate specific differences.
