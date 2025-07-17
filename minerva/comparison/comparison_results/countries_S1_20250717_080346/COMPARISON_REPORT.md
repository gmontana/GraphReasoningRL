# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_080346
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.2917 | 0.7083 | 70.83% | ✗ |
| hits@3 | 1.0000 | 0.7917 | 0.2083 | 20.83% | ✗ |
| hits@5 | 1.0000 | 0.7917 | 0.2083 | 20.83% | ✗ |
| hits@10 | 1.0000 | 0.7917 | 0.2083 | 20.83% | ✗ |
| mrr | 1.0000 | 0.5417 | 0.4583 | 45.83% | ✗ |
