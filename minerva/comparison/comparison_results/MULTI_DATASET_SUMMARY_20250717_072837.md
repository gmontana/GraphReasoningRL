# MINERVA Multi-Dataset Comparison Summary

⚠️ **IMPORTANT**: Comparing TRAINING VALIDATION metrics only

**Date**: 20250717_072837
**Datasets Tested**: 5

## Summary Table

| Dataset | Hits@10 (TF) | Hits@10 (PyTorch) | MRR (TF) | MRR (PyTorch) | Status |
|---------|--------------|-------------------|----------|---------------|--------|
| countries_S1 | 1.0000 | 0.6667 | 1.0000 | 0.5208 | ✗ Differ |
| countries_S2 | 0.6250 | 0.4583 | 0.4927 | 0.1487 | ✗ Differ |
| countries_S3 | 0.9583 | 0.5000 | 0.9583 | 0.3514 | ✗ Differ |
| kinship | 0.0599 | 0.1124 | 0.0275 | 0.0343 | ✓ Match |
| umls | 0.2485 | 0.1641 | 0.0718 | 0.0590 | ✗ Differ |

**Overall Matching Rate**: 1/5 datasets

## Per-Metric Summary

- **hits@1**: 2/5 datasets match
- **hits@3**: 2/5 datasets match
- **hits@5**: 2/5 datasets match
- **hits@10**: 1/5 datasets match
- **mrr**: 2/5 datasets match
