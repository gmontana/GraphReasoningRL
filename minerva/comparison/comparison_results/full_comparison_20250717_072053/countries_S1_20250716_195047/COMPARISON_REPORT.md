# MINERVA Implementation Comparison Report

**Dataset**: countries_S1
**Iterations**: 100
**Random Seed**: 42
**Timestamp**: 20250716_195047
## Results Comparison

| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |
|--------|------------|---------|------------|---------------|--------|
| hits@1 | 1.0000 | 0.4583 | 0.5417 | 54.17% | ✗ |
| hits@3 | 1.0000 | 0.9583 | 0.0417 | 4.17% | ✓ |
| hits@5 | 1.0000 | 0.9583 | 0.0417 | 4.17% | ✓ |
| hits@10 | 1.0000 | 1.0000 | 0.0000 | 0.00% | ✓ |
| mrr | 0.0000 | 0.6847 | 0.6847 | 68470.00% | ✗ |
