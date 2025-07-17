# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S3
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072710
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.9583 | 0.2500 | 0.7083 | 73.91% | ✗ |
| hits@3 | 0.9583 | 0.4583 | 0.5000 | 52.18% | ✗ |
| hits@5 | 0.9583 | 0.4583 | 0.5000 | 52.18% | ✗ |
| hits@10 | 0.9583 | 0.5000 | 0.4583 | 47.82% | ✗ |
| mrr | 0.9583 | 0.3514 | 0.6069 | 63.33% | ✗ |
