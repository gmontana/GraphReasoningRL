# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: kinship
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_080428
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0122 | 0.0084 | 0.0038 | 31.15% | ✓ |
| hits@3 | 0.0309 | 0.0281 | 0.0028 | 9.06% | ✓ |
| hits@5 | 0.0478 | 0.0421 | 0.0057 | 11.92% | ✓ |
| hits@10 | 0.0890 | 0.0880 | 0.0010 | 1.12% | ✓ |
| mrr | 0.0334 | 0.0311 | 0.0023 | 6.89% | ✓ |
