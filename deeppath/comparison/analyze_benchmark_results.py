#!/usr/bin/env python
"""
Script to analyze and compare benchmark results between PyTorch and TensorFlow implementations.
"""

import sys
import os
import numpy as np
from collections import defaultdict

# Get the script directory and repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def parse_path_stats(filepath):
    """Parse path stats file into a dictionary of path: count"""
    stats = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        path, count = parts
                        stats[path] = int(count)
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath}")
    return stats

def parse_path_to_use(filepath):
    """Parse path_to_use file into a list of paths"""
    paths = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    paths.append(line.strip())
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath}")
    return paths

def extract_success_rate(log_filepath):
    """Extract success rate from log file"""
    try:
        with open(log_filepath, 'r') as f:
            for line in f:
                if 'Success rate:' in line:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        rate_str = parts[-1].strip()
                        rate_str = rate_str.rstrip('%')
                        try:
                            return float(rate_str) / 100.0
                        except ValueError:
                            pass
    except FileNotFoundError:
        print(f"Warning: Could not find {log_filepath}")
    return None

def extract_runtime(log_filepath):
    """Extract runtime from log file"""
    try:
        with open(log_filepath, 'r') as f:
            for line in f:
                if 'runtime:' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        time_str = parts[-1].strip()
                        try:
                            return float(time_str.split()[0])
                        except ValueError:
                            pass
    except FileNotFoundError:
        print(f"Warning: Could not find {log_filepath}")
    return None

def compare_path_distributions(pytorch_stats, tensorflow_stats):
    """Compare path distributions between PyTorch and TensorFlow"""
    all_paths = set(list(pytorch_stats.keys()) + list(tensorflow_stats.keys()))
    
    comparison = []
    for path in all_paths:
        pt_count = pytorch_stats.get(path, 0)
        tf_count = tensorflow_stats.get(path, 0)
        comparison.append((path, pt_count, tf_count, abs(pt_count - tf_count)))
    
    # Sort by absolute difference
    comparison.sort(key=lambda x: x[3], reverse=True)
    return comparison

def analyze_relation(relation):
    """Analyze benchmark results for a specific relation"""
    print(f"\n=== Analyzing results for relation: {relation} ===\n")
    
    # File paths
    pytorch_log = os.path.join(SCRIPT_DIR, f"logs/pytorch_{relation}.log")
    tensorflow_log = os.path.join(SCRIPT_DIR, f"logs/tensorflow_{relation}.log")
    pytorch_path_stats = os.path.join(REPO_ROOT, f"NELL-995/tasks/{relation}/path_stats_pytorch.txt")
    tensorflow_path_stats = os.path.join(REPO_ROOT, f"NELL-995/tasks/{relation}/path_stats_tensorflow.txt")
    pytorch_path_to_use = os.path.join(REPO_ROOT, f"NELL-995/tasks/{relation}/path_to_use_pytorch.txt")
    tensorflow_path_to_use = os.path.join(REPO_ROOT, f"NELL-995/tasks/{relation}/path_to_use_tensorflow.txt")
    
    # Extract metrics
    pt_success_rate = extract_success_rate(pytorch_log)
    tf_success_rate = extract_success_rate(tensorflow_log)
    pt_runtime = extract_runtime(pytorch_log)
    tf_runtime = extract_runtime(tensorflow_log)
    
    # Parse path stats
    pt_path_stats = parse_path_stats(pytorch_path_stats)
    tf_path_stats = parse_path_stats(tensorflow_path_stats)
    
    # Parse path to use
    pt_paths = parse_path_to_use(pytorch_path_to_use)
    tf_paths = parse_path_to_use(tensorflow_path_to_use)
    
    # Print performance metrics
    print("Performance Metrics:")
    if pt_success_rate is not None:
        print(f"PyTorch Success Rate: {pt_success_rate*100:.2f}%")
    else:
        print("PyTorch Success Rate: Not available")
        
    if tf_success_rate is not None:
        print(f"TensorFlow Success Rate: {tf_success_rate*100:.2f}%")
    else:
        print("TensorFlow Success Rate: Not available")
        
    if pt_success_rate and tf_success_rate:
        diff = (pt_success_rate - tf_success_rate) * 100
        print(f"Difference: {diff:.2f}% ({'+' if diff > 0 else ''}{diff:.2f}%)")
    
    print(f"\nRuntime:")
    if pt_runtime is not None:
        print(f"PyTorch: {pt_runtime:.2f}s")
    else:
        print("PyTorch runtime: Not available")
        
    if tf_runtime is not None:
        print(f"TensorFlow: {tf_runtime:.2f}s")
    else:
        print("TensorFlow runtime: Not available")
        
    if pt_runtime and tf_runtime:
        speedup = tf_runtime / pt_runtime
        print(f"Speedup: {speedup:.2f}x")
    
    # Compare path distributions if available
    if pt_path_stats and tf_path_stats:
        path_comparison = compare_path_distributions(pt_path_stats, tf_path_stats)
        
        print("\nPath Distribution Differences (top 5):")
        for i, (path, pt_count, tf_count, diff) in enumerate(path_comparison[:5]):
            print(f"{i+1}. Path: {path}")
            print(f"   PyTorch: {pt_count} | TensorFlow: {tf_count} | Diff: {diff}")
    else:
        print("\nPath distribution comparison not available: Missing path statistics")
    
    # Compare top paths if available
    if pt_paths and tf_paths:
        common_paths = set(pt_paths) & set(tf_paths)
        pt_unique = set(pt_paths) - set(tf_paths)
        tf_unique = set(tf_paths) - set(pt_paths)
        
        print(f"\nPath Overlap Analysis:")
        print(f"Common Paths: {len(common_paths)} ({len(common_paths)/max(len(pt_paths), len(tf_paths))*100:.2f}%)")
        print(f"PyTorch Unique Paths: {len(pt_unique)}")
        print(f"TensorFlow Unique Paths: {len(tf_unique)}")
        
        # Compare top 3 paths
        print("\nTop Paths Comparison:")
        print("PyTorch Top 3:")
        for i, path in enumerate(pt_paths[:min(3, len(pt_paths))]):
            print(f"{i+1}. {path}")
        
        print("\nTensorFlow Top 3:")
        for i, path in enumerate(tf_paths[:min(3, len(tf_paths))]):
            print(f"{i+1}. {path}")
    else:
        print("\nPath overlap analysis not available: Missing path data")
    
    # Save analysis to file
    analysis_file = os.path.join(SCRIPT_DIR, f"logs/analysis_{relation}.txt")
    try:
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
        with open(analysis_file, 'w') as f:
            f.write(f"=== Analysis Results for {relation} ===\n\n")
            f.write("Performance Metrics:\n")
            if pt_success_rate is not None:
                f.write(f"PyTorch Success Rate: {pt_success_rate*100:.2f}%\n")
            else:
                f.write("PyTorch Success Rate: Not available\n")
                
            if tf_success_rate is not None:
                f.write(f"TensorFlow Success Rate: {tf_success_rate*100:.2f}%\n")
            else:
                f.write("TensorFlow Success Rate: Not available\n")
                
            if pt_success_rate and tf_success_rate:
                diff = (pt_success_rate - tf_success_rate) * 100
                f.write(f"Difference: {diff:.2f}% ({'+' if diff > 0 else ''}{diff:.2f}%)\n")
            
            f.write(f"\nRuntime:\n")
            if pt_runtime is not None:
                f.write(f"PyTorch: {pt_runtime:.2f}s\n")
            else:
                f.write("PyTorch runtime: Not available\n")
                
            if tf_runtime is not None:
                f.write(f"TensorFlow: {tf_runtime:.2f}s\n")
            else:
                f.write("TensorFlow runtime: Not available\n")
                
            if pt_runtime and tf_runtime:
                speedup = tf_runtime / pt_runtime
                f.write(f"Speedup: {speedup:.2f}x\n")
            
            if pt_paths and tf_paths:
                f.write("\nPath Overlap Analysis:\n")
                f.write(f"Common Paths: {len(common_paths)} ({len(common_paths)/max(len(pt_paths), len(tf_paths))*100:.2f}%)\n")
                f.write(f"PyTorch Unique Paths: {len(pt_unique)}\n")
                f.write(f"TensorFlow Unique Paths: {len(tf_unique)}\n")
        
        print(f"\nAnalysis saved to {analysis_file}")
    except Exception as e:
        print(f"Error saving analysis to file: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        relation = sys.argv[1]
        analyze_relation(relation)
    else:
        # Default to athletePlaysForTeam if no relation specified
        analyze_relation("athletePlaysForTeam")

if __name__ == "__main__":
    main()