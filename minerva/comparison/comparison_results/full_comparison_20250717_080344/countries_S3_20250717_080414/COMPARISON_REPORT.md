# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S3
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_080414
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.8333 | 0.2917 | 0.5416 | 64.99% | ✗ |
| hits@3 | 0.8750 | 0.4167 | 0.4583 | 52.38% | ✗ |
| hits@5 | 0.8750 | 0.4167 | 0.4583 | 52.38% | ✗ |
| hits@10 | 0.8750 | 0.4167 | 0.4583 | 52.38% | ✗ |
| mrr | 0.8542 | 0.3542 | 0.5000 | 58.53% | ✗ |
