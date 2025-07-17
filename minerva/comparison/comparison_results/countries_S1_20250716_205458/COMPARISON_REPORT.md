# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S1
**Iterations**: 10
**Random Seed**: 42
**Timestamp**: 20250716_205458
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.5000 | 0.5000 | 50.00% | ✗ |
| hits@3 | 1.0000 | 0.9583 | 0.0417 | 4.17% | ✓ |
| hits@5 | 1.0000 | 0.9583 | 0.0417 | 4.17% | ✓ |
| hits@10 | 1.0000 | 0.9583 | 0.0417 | 4.17% | ✓ |
| mrr | 1.0000 | 0.7292 | 0.2708 | 27.08% | ✗ |
