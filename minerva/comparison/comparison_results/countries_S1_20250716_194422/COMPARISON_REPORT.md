# MINERVA Implementation Comparison Report

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250716_194422
**Verdict**: FAIL

## Results Comparison

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.7083 | 0.2917 | 29.17% | ✗ |
| hits@3 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| hits@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| hits@10 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| mrr | 0.0000 | 0.8403 | 0.8403 | 84030.00% | ✗ |

## Analysis

The implementations show significant differences in results.
Possible causes:
- Random initialization differences
- Numerical precision differences between frameworks
- Implementation bugs
- Different random sampling behavior

Recommendation: Run with more iterations or investigate specific differences.
