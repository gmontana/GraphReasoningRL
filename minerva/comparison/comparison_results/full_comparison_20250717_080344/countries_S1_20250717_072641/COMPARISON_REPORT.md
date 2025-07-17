# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072641
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.3750 | 0.6250 | 62.50% | ✗ |
| hits@3 | 1.0000 | 0.6667 | 0.3333 | 33.33% | ✗ |
| hits@5 | 1.0000 | 0.6667 | 0.3333 | 33.33% | ✗ |
| hits@10 | 1.0000 | 0.6667 | 0.3333 | 33.33% | ✗ |
| mrr | 1.0000 | 0.5208 | 0.4792 | 47.92% | ✗ |
