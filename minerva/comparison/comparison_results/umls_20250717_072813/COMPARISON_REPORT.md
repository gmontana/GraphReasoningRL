# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: umls
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_072813
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0123 | 0.0215 | 0.0092 | 74.80% | ✓ |
| hits@3 | 0.0752 | 0.0629 | 0.0123 | 16.36% | ✓ |
| hits@5 | 0.1396 | 0.0828 | 0.0568 | 40.69% | ✓ |
| hits@10 | 0.2485 | 0.1641 | 0.0844 | 33.96% | ✗ |
| mrr | 0.0718 | 0.0590 | 0.0128 | 17.83% | ✓ |
