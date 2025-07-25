# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S3
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072120
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.7917 | 0.0000 | 0.7917 | 100.00% | ✗ |
| hits@3 | 0.8750 | 0.0000 | 0.8750 | 100.00% | ✗ |
| hits@5 | 0.8750 | 0.0000 | 0.8750 | 100.00% | ✗ |
| hits@10 | 0.8750 | 0.0000 | 0.8750 | 100.00% | ✗ |
| mrr | 0.8333 | 0.0000 | 0.8333 | 100.00% | ✗ |
