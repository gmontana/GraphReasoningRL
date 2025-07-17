# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: countries_S1
**Iterations**: 50
**Random Seed**: 42
**Timestamp**: 20250717_072442
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.2917 | 0.7083 | 70.83% | ✗ |
| hits@3 | 1.0000 | 0.8750 | 0.1250 | 12.50% | ✓ |
| hits@5 | 1.0000 | 0.8750 | 0.1250 | 12.50% | ✓ |
| hits@10 | 1.0000 | 0.8750 | 0.1250 | 12.50% | ✓ |
| mrr | 1.0000 | 0.5764 | 0.4236 | 42.36% | ✗ |
