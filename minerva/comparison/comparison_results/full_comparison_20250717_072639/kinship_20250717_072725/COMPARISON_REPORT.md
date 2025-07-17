# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: kinship
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072725
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0131 | 0.0075 | 0.0056 | 42.75% | ✓ |
| hits@3 | 0.0281 | 0.0281 | 0.0000 | 0.00% | ✓ |
| hits@5 | 0.0375 | 0.0506 | 0.0131 | 34.93% | ✓ |
| hits@10 | 0.0599 | 0.1124 | 0.0525 | 87.65% | ✓ |
| mrr | 0.0275 | 0.0343 | 0.0068 | 24.73% | ✓ |
