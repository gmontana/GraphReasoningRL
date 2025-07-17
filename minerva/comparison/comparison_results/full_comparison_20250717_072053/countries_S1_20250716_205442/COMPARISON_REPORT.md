# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250716_205442
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.9583 | 0.0417 | 4.17% | ✓ |
| hits@3 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| hits@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| hits@10 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| mrr | 1.0000 | 0.9792 | 0.0208 | 2.08% | ✓ |
