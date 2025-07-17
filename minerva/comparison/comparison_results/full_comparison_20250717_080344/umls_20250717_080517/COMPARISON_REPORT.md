# MINERVA Implementation Comparison Report

⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics
⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues

**Dataset**: umls
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250717_080517
## Results Comparison (Training Validation Metrics Only)

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 0.0583 | 0.0138 | 0.0445 | 76.33% | ✓ |
| hits@3 | 0.1396 | 0.0445 | 0.0951 | 68.12% | ✗ |
| hits@5 | 0.1887 | 0.0844 | 0.1043 | 55.27% | ✗ |
| hits@10 | 0.3006 | 0.1656 | 0.1350 | 44.91% | ✗ |
| mrr | 0.1229 | 0.0528 | 0.0701 | 57.04% | ✗ |
