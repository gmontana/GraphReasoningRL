#!/bin/bash
# Script to run comprehensive multi-dataset comparison between TensorFlow and PyTorch MINERVA implementations

echo "====================================================================="
echo "MINERVA Implementation Multi-Dataset Comparison"
echo "====================================================================="
echo ""
echo "This script will compare TensorFlow and PyTorch implementations on multiple datasets."
echo "Note: Only training validation metrics are compared due to TF checkpoint issues."
echo ""

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="comparison_results/full_comparison_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

# Log file
LOG_FILE="$RESULTS_DIR/comparison_log.txt"

echo "Results will be saved to: $RESULTS_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Start logging
{
    echo "Multi-Dataset Comparison Started at $(date)"
    echo "====================================================================="
    
    # Run comparison on multiple datasets
    # Using fewer iterations for faster testing, increase for more thorough comparison
    python compare.py --test_multiple --iterations 100 --seed 42
    
    echo ""
    echo "====================================================================="
    echo "Comparison completed at $(date)"
    
    # Copy all individual results to summary directory
    echo "Copying individual results..."
    cp -r comparison_results/countries_S1_* $RESULTS_DIR/ 2>/dev/null
    cp -r comparison_results/countries_S2_* $RESULTS_DIR/ 2>/dev/null
    cp -r comparison_results/countries_S3_* $RESULTS_DIR/ 2>/dev/null
    cp -r comparison_results/kinship_* $RESULTS_DIR/ 2>/dev/null
    cp -r comparison_results/umls_* $RESULTS_DIR/ 2>/dev/null
    
    # Copy multi-dataset summary
    cp comparison_results/MULTI_DATASET_SUMMARY_*.md $RESULTS_DIR/ 2>/dev/null
    
    echo "All results copied to $RESULTS_DIR"
    
} | tee $LOG_FILE

echo ""
echo "====================================================================="
echo "Comparison complete! Check the results in:"
echo "  $RESULTS_DIR"
echo ""
echo "Key files to review:"
echo "  - MULTI_DATASET_SUMMARY_*.md - Overall comparison summary"
echo "  - */COMPARISON_REPORT.md - Individual dataset reports"
echo "  - comparison_log.txt - Full execution log"
echo "====================================================================="