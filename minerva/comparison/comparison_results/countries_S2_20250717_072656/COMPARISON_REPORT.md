# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S2
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072656
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.4583 | 0.0417 | 0.4166 | 90.90% | ✗ |
| hits@3 | 0.5000 | 0.2083 | 0.2917 | 58.34% | ✗ |
| hits@5 | 0.5417 | 0.3750 | 0.1667 | 30.77% | ✗ |
| hits@10 | 0.6250 | 0.4583 | 0.1667 | 26.67% | ✗ |
| mrr | 0.4927 | 0.1487 | 0.3440 | 69.82% | ✗ |
