# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_080323
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.3333 | 0.6667 | 66.67% | ✗ |
| hits@3 | 1.0000 | 0.5833 | 0.4167 | 41.67% | ✗ |
| hits@5 | 1.0000 | 0.6250 | 0.3750 | 37.50% | ✗ |
| hits@10 | 1.0000 | 0.6250 | 0.3750 | 37.50% | ✗ |
| mrr | 1.0000 | 0.4688 | 0.5312 | 53.12% | ✗ |
