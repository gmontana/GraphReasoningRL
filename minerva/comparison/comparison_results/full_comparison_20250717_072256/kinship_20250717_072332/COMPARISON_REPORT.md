# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: kinship
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072332
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0169 | 0.0000 | 0.0169 | 100.00% | ✗ |
| hits@3 | 0.0393 | 0.0000 | 0.0393 | 100.00% | ✗ |
| hits@5 | 0.0543 | 0.0000 | 0.0543 | 100.00% | ✗ |
| hits@10 | 0.1021 | 0.0000 | 0.1021 | 100.00% | ✗ |
| mrr | 0.0394 | 0.0000 | 0.0394 | 100.00% | ✗ |
