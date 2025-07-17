#!/usr/bin/env python
"""
Compare TensorFlow and PyTorch MINERVA implementations
This script runs both implementations on the same dataset and compares results

IMPORTANT: This comparison only evaluates TRAINING PHASE performance.
- TensorFlow: Uses training validation metrics (dev set evaluation during training)
- PyTorch: Uses training validation metrics (dev set evaluation during training)
- Test phase evaluation is NOT included due to TensorFlow checkpoint compatibility issues
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np
from datetime import datetime

def setup_environments():
    """Set up virtual environments for TF and PyTorch"""
    print("=" * 80)
    print("MINERVA Implementation Comparison")
    print("=" * 80)
    print("⚠️  WARNING: This comparison only evaluates TRAINING PHASE performance")
    print("⚠️  Test phase evaluation is skipped due to TensorFlow compatibility issues")
    print("=" * 80)
    
    # Check if we can import both frameworks
    tf_available = False
    pytorch_available = False
    
    try:
        import tensorflow as tf
        tf_available = True
        print(f"✓ TensorFlow {tf.__version__} available")
    except ImportError:
        print("✗ TensorFlow not available")
    
    try:
        import torch
        pytorch_available = True
        print(f"✓ PyTorch {torch.__version__} available")
    except ImportError:
        print("✗ PyTorch not available")
    
    if not tf_available or not pytorch_available:
        print("\nPlease install required packages:")
        if not tf_available:
            print("  pip install tensorflow==1.15.0")
        if not pytorch_available:
            print("  pip install torch")
        return False
    
    return True


def run_comparison(dataset='countries_S1', iterations=100, seed=42):
    """Run both implementations and compare results"""
    
    if not setup_environments():
        return
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join(base_dir, f'comparison_results/{dataset}_{timestamp}')
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"\nRunning comparison on dataset: {dataset}")
    print(f"Iterations: {iterations}")
    print(f"Random seed: {seed}")
    print(f"Output directory: {comparison_dir}")
    print(f"\n⚠️  IMPORTANT: Comparing TRAINING VALIDATION metrics only")
    print(f"⚠️  TensorFlow test phase is disabled due to checkpoint issues")
    
    # Run TensorFlow implementation
    print("\n" + "="*60)
    print("Running TensorFlow Implementation (Training Validation Only)")
    print("="*60)
    
    tf_cmd = [
        'python', 'run_tensorflow.py',
        '--dataset', dataset,
        '--iterations', str(iterations)
    ]
    
    tf_env = os.environ.copy()
    tf_env['PYTHONHASHSEED'] = str(seed)
    tf_env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("Running TensorFlow implementation...")
    tf_result = subprocess.run(tf_cmd, env=tf_env)
    
    # Parse TF results
    tf_results_file = os.path.join(base_dir, f'outputs_tensorflow/{dataset}/results.json')
    if os.path.exists(tf_results_file):
        with open(tf_results_file, 'r') as f:
            tf_results = json.load(f)
    else:
        print("TensorFlow run failed!")
        print(tf_result.stderr)
        return
    
    # Run PyTorch implementation
    print("\n" + "="*60)
    print("Running PyTorch Implementation (Training Validation Only)")
    print("="*60)
    
    pytorch_cmd = [
        'python', 'run_pytorch.py',
        '--dataset', dataset,
        '--iterations', str(iterations)
    ]
    
    pytorch_env = os.environ.copy()
    pytorch_env['PYTHONHASHSEED'] = str(seed)
    
    print("Running PyTorch implementation...")
    pytorch_result = subprocess.run(pytorch_cmd, env=pytorch_env)
    
    # Parse PyTorch results
    pytorch_results_file = os.path.join(base_dir, f'outputs_pytorch/{dataset}/results.json')
    if os.path.exists(pytorch_results_file):
        with open(pytorch_results_file, 'r') as f:
            pytorch_results = json.load(f)
    else:
        print("PyTorch run failed!")
        print(pytorch_result.stderr)
        return
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison Results (Training Validation Metrics Only)")
    print("="*60)
    
    comparison = {
        'dataset': dataset,
        'iterations': iterations,
        'seed': seed,
        'timestamp': timestamp,
        'tensorflow_results': tf_results,
        'pytorch_results': pytorch_results,
        'comparison': {}
    }
    
    # Calculate differences
    print(f"\n{'Metric':<15} {'TensorFlow':<12} {'PyTorch':<12} {'Difference':<12} {'Status'}")
    print("-" * 60)
    
    all_close = True
    major_differences = 0
    
    for metric in ['hits@1', 'hits@3', 'hits@5', 'hits@10', 'mrr']:
        tf_val = tf_results['metrics'].get(metric, 0.0)
        pt_val = pytorch_results['metrics'].get(metric, 0.0)
        diff = abs(tf_val - pt_val)
        
        # Skip comparison if one implementation doesn't have the metric
        if tf_val == 0.0 and pt_val == 0.0:
            is_close = True
            status = "- N/A"
        elif tf_val == 0.0 or pt_val == 0.0:
            # If one is missing, this is a problem but don't count as major difference
            is_close = False
            status = "- Missing"
        else:
            # Use reasonable tolerance for RL algorithms
            is_close = np.isclose(tf_val, pt_val, rtol=0.10, atol=0.05)
            status = "✓ Close" if is_close else "✗ Different"
        
        if not is_close and tf_val != 0.0 and pt_val != 0.0:
            major_differences += 1
        
        comparison['comparison'][metric] = {
            'tensorflow': float(tf_val),
            'pytorch': float(pt_val),
            'difference': float(diff),
            'relative_diff': float(diff / max(tf_val, 0.001)),
            'is_close': bool(is_close)
        }
        
        print(f"{metric:<15} {tf_val:<12.4f} {pt_val:<12.4f} {diff:<12.4f} {status}")
    
    print("="*60)
    
    # Save comparison results
    comparison_file = os.path.join(comparison_dir, 'comparison_results.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nDetailed comparison saved to: {comparison_file}")
    
    # Create summary report
    create_summary_report(comparison, comparison_dir)
    
    return comparison


def create_summary_report(comparison, output_dir):
    """Create a markdown summary report"""
    
    report_file = os.path.join(output_dir, 'COMPARISON_REPORT.md')
    
    with open(report_file, 'w') as f:
        f.write("# MINERVA Implementation Comparison Report\n\n")
        f.write("⚠️ **IMPORTANT**: This comparison only evaluates TRAINING VALIDATION metrics\n")
        f.write("⚠️ **Test phase evaluation is disabled** due to TensorFlow checkpoint compatibility issues\n\n")
        f.write(f"**Dataset**: {comparison['dataset']}\n")
        f.write(f"**Iterations**: {comparison['iterations']}\n")
        f.write(f"**Random Seed**: {comparison['seed']}\n")
        f.write(f"**Timestamp**: {comparison['timestamp']}\n")
        
        f.write("## Results Comparison (Training Validation Metrics Only)\n\n")
        f.write("| Metric | TensorFlow | PyTorch | Difference | Relative Diff | Status |\n")
        f.write("|--------|------------|---------|------------|---------------|--------|\n")
        
        for metric, values in comparison['comparison'].items():
            status = "✓" if values['is_close'] else "✗"
            f.write(f"| {metric} | {values['tensorflow']:.4f} | {values['pytorch']:.4f} | "
                   f"{values['difference']:.4f} | {values['relative_diff']:.2%} | {status} |\n")
        
    
    print(f"Summary report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare TF and PyTorch MINERVA implementations')
    parser.add_argument('--dataset', default='countries_S1', 
                        help='Dataset to use (e.g., countries_S1, kinship, WN18RR)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Multiple datasets to compare')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--test_multiple', action='store_true',
                        help='Test on multiple datasets')
    
    args = parser.parse_args()
    
    if args.datasets:
        # Use specified datasets
        datasets = args.datasets
    elif args.test_multiple:
        # Default set of datasets for multiple testing
        datasets = ['countries_S1', 'countries_S2', 'countries_S3', 'kinship', 'umls']
    else:
        # Single dataset
        datasets = [args.dataset]
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Testing dataset: {dataset}")
        print('='*80)
        
        result = run_comparison(dataset, args.iterations, args.seed)
        if result:
            all_results.append(result)
    
    # Create overall summary if multiple datasets
    if len(datasets) > 1:
        create_overall_summary(all_results)
    
    
def create_overall_summary(all_results):
    """Create summary report for multiple datasets"""
    print("\n" + "="*80)
    print("OVERALL COMPARISON SUMMARY")
    print("="*80)
    
    summary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comparison_results')
    os.makedirs(summary_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(summary_dir, f'MULTI_DATASET_SUMMARY_{timestamp}.md')
    
    with open(summary_file, 'w') as f:
        f.write("# MINERVA Multi-Dataset Comparison Summary\n\n")
        f.write("⚠️ **IMPORTANT**: Comparing TRAINING VALIDATION metrics only\n\n")
        f.write(f"**Date**: {timestamp}\n")
        f.write(f"**Datasets Tested**: {len(all_results)}\n\n")
        
        f.write("## Summary Table\n\n")
        f.write("| Dataset | Hits@10 (TF) | Hits@10 (PyTorch) | MRR (TF) | MRR (PyTorch) | Status |\n")
        f.write("|---------|--------------|-------------------|----------|---------------|--------|\n")
        
        total_close = 0
        for result in all_results:
            comp = result['comparison']
            h10_close = comp['hits@10']['is_close']
            mrr_close = comp['mrr']['is_close']
            overall_close = h10_close and mrr_close
            
            if overall_close:
                total_close += 1
                status = "✓ Match"
            else:
                status = "✗ Differ"
            
            f.write(f"| {result['dataset']} | "
                   f"{comp['hits@10']['tensorflow']:.4f} | "
                   f"{comp['hits@10']['pytorch']:.4f} | "
                   f"{comp['mrr']['tensorflow']:.4f} | "
                   f"{comp['mrr']['pytorch']:.4f} | "
                   f"{status} |\n")
        
        f.write(f"\n**Overall Matching Rate**: {total_close}/{len(all_results)} datasets\n")
        
        # Per-metric summary
        f.write("\n## Per-Metric Summary\n\n")
        metrics = ['hits@1', 'hits@3', 'hits@5', 'hits@10', 'mrr']
        
        for metric in metrics:
            close_count = sum(1 for r in all_results if r['comparison'][metric]['is_close'])
            f.write(f"- **{metric}**: {close_count}/{len(all_results)} datasets match\n")
    
    print(f"\nMulti-dataset summary saved to: {summary_file}")
    print(f"Overall matching rate: {total_close}/{len(all_results)} datasets")


if __name__ == '__main__':
    main()