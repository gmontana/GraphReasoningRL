# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S2
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_080400
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.3750 | 0.0000 | 0.3750 | 100.00% | ✗ |
| hits@3 | 0.4583 | 0.0833 | 0.3750 | 81.82% | ✗ |
| hits@5 | 0.4583 | 0.1667 | 0.2916 | 63.63% | ✗ |
| hits@10 | 0.5000 | 0.2083 | 0.2917 | 58.34% | ✗ |
| mrr | 0.4139 | 0.0604 | 0.3535 | 85.41% | ✗ |
