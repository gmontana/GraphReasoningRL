# MINERVA Implementation Comparison

This directory contains tools to compare the PyTorch and TensorFlow implementations of MINERVA.

## Directory Structure

```
comparison/
├── compare.py                      # Main comparison script for single datasets
├── run_multi_dataset_comparison.sh # Batch comparison across multiple datasets
├── run_pytorch.py                  # Standalone PyTorch runner
├── run_tensorflow.py               # Standalone TensorFlow runner
├── tf_patched/                     # TensorFlow 2.x compatible version
├── outputs_pytorch/                # PyTorch execution results
├── outputs_tensorflow/             # TensorFlow execution results
├── comparison_results/             # Comparison reports and summaries
├── COMPARISON_GUIDE.md             # Detailed usage guide
└── PARITY_REPORT.md               # Feature parity analysis
```

## Quick Start

```bash
# Compare both implementations on a single dataset
python compare.py --dataset countries_S1 --iterations 50

# Run comparison on multiple datasets
./run_multi_dataset_comparison.sh

# Run individual implementations
python run_pytorch.py --dataset countries_S1 --iterations 50
python run_tensorflow.py --dataset countries_S1 --iterations 50
```

## Important Notes

⚠️ **TensorFlow Limitations**: The TensorFlow implementation currently only supports training validation metrics due to checkpoint restoration issues in the test phase.

## Results

The comparison results are stored in `comparison_results/` with:
- Individual dataset comparisons with timestamps
- Multi-dataset summary reports
- Latest full comparison in `full_comparison_*/`

## Expected Results

PyTorch implementation typically achieves:
- Hits@1: 0.3-0.7
- Hits@3: 0.6-0.9
- Hits@5: 0.7-0.9
- Hits@10: 0.7-0.9
- MRR: 0.4-0.8

Note: TensorFlow training validation metrics may show inflated values (often perfect scores).