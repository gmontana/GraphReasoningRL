# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: kinship
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072131
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0487 | 0.0000 | 0.0487 | 100.00% | ✗ |
| hits@3 | 0.0964 | 0.0000 | 0.0964 | 100.00% | ✗ |
| hits@5 | 0.1348 | 0.0000 | 0.1348 | 100.00% | ✗ |
| hits@10 | 0.2041 | 0.0000 | 0.2041 | 100.00% | ✗ |
| mrr | 0.0932 | 0.0000 | 0.0932 | 100.00% | ✗ |
