# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: umls
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072345
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0061 | 0.0000 | 0.0061 | 100.00% | ✗ |
| hits@3 | 0.0491 | 0.0000 | 0.0491 | 100.00% | ✗ |
| hits@5 | 0.0752 | 0.0000 | 0.0752 | 100.00% | ✗ |
| hits@10 | 0.1549 | 0.0000 | 0.1549 | 100.00% | ✗ |
| mrr | 0.0434 | 0.0000 | 0.0434 | 100.00% | ✗ |
