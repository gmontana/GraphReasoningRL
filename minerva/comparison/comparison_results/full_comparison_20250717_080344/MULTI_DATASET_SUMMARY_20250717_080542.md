# MINERVA Multi-Dataset Comparison Summary

⚠️ **IMPORTANT**: Comparing TRAINING VALIDATION metrics only

**Date**: 20250717_080542
**Datasets Tested**: 5

## Summary Table

| Dataset | Hits@10 (TF) | Hits@10 (PyTorch) | MRR (TF) | MRR (PyTorch) | Status |
|---------|--------------|-------------------|----------|---------------|--------|
| countries_S1 | 1.0000 | 0.7917 | 1.0000 | 0.5417 | ✗ Differ |
| countries_S2 | 0.5000 | 0.2083 | 0.4139 | 0.0604 | ✗ Differ |
| countries_S3 | 0.8750 | 0.4167 | 0.8542 | 0.3542 | ✗ Differ |
| kinship | 0.0890 | 0.0880 | 0.0334 | 0.0311 | ✓ Match |
| umls | 0.3006 | 0.1656 | 0.1229 | 0.0528 | ✗ Differ |

**Overall Matching Rate**: 1/5 datasets

## Per-Metric Summary

- **hits@1**: 2/5 datasets match
- **hits@3**: 1/5 datasets match
- **hits@5**: 1/5 datasets match
- **hits@10**: 1/5 datasets match
- **mrr**: 1/5 datasets match
