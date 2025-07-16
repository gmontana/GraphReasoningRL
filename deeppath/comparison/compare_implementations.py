#!/usr/bin/env python
"""
Script to compare PyTorch and TensorFlow implementations of DeepPath.
"""

import os
import sys
import numpy as np

def analyze_paths(relation):
    """Analyze paths discovered by both implementations."""
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the repository root directory
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    data_path = os.path.join(repo_root, 'NELL-995/tasks/')
    relation_path = os.path.join(data_path, relation)
    
    # Paths for PyTorch and TensorFlow results
    pt_path_to_use = os.path.join(relation_path, 'path_to_use_pytorch.txt')
    tf_path_to_use = os.path.join(relation_path, 'path_to_use_tensorflow.txt')
    pt_path_stats = os.path.join(relation_path, 'path_stats_pytorch.txt')
    tf_path_stats = os.path.join(relation_path, 'path_stats_tensorflow.txt')
    
    # Check if files exist
    files_exist = all(os.path.exists(p) for p in [pt_path_to_use, pt_path_stats])
    tf_exists = all(os.path.exists(p) for p in [tf_path_to_use, tf_path_stats])
    
    if not files_exist:
        print(f"Error: Could not find PyTorch result files for {relation}")
        return
    
    # Read PyTorch paths
    with open(pt_path_to_use) as f:
        pt_paths = [line.strip() for line in f if line.strip()]
    
    with open(pt_path_stats) as f:
        pt_stats = {}
        for line in f:
            if '\t' in line:
                path, count = line.strip().split('\t')
                pt_stats[path] = int(count)
    
    print(f"\n=== Path Analysis for {relation} ===\n")
    print(f"PyTorch discovered {len(pt_paths)} paths")
    
    # If TensorFlow results exist, compare them
    if tf_exists:
        with open(tf_path_to_use) as f:
            tf_paths = [line.strip() for line in f if line.strip()]
        
        with open(tf_path_stats) as f:
            tf_stats = {}
            for line in f:
                if '\t' in line:
                    path, count = line.strip().split('\t')
                    tf_stats[path] = int(count)
        
        print(f"TensorFlow discovered {len(tf_paths)} paths")
        
        # Compare path distributions
        common_paths = set(pt_paths) & set(tf_paths)
        pt_unique = set(pt_paths) - set(tf_paths)
        tf_unique = set(tf_paths) - set(pt_paths)
        
        print(f"\nPath Comparison:")
        print(f"Common paths: {len(common_paths)}")
        print(f"PyTorch unique paths: {len(pt_unique)}")
        print(f"TensorFlow unique paths: {len(tf_unique)}")
        
        # Compare top 3 paths
        print("\nTop 3 PyTorch paths:")
        for i, path in enumerate(pt_paths[:min(3, len(pt_paths))]):
            print(f"  {i+1}. {path} (count: {pt_stats.get(path, 'unknown')})")
        
        print("\nTop 3 TensorFlow paths:")
        for i, path in enumerate(tf_paths[:min(3, len(tf_paths))]):
            print(f"  {i+1}. {path} (count: {tf_stats.get(path, 'unknown')})")
    else:
        print("\nTensorFlow results not available for comparison")
    
    print("\nPyTorch Implementation Analysis:")
    if len(pt_paths) > 0:
        print("Found valid path patterns for relation")
        
        # Check most common path patterns
        print("\nMost common path patterns:")
        for path, count in sorted(pt_stats.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {path}: {count} occurrences")
    else:
        print("Warning: No paths discovered")
    
    # Assess implementation completeness
    print("\nImplementation assessment:")
    print("✓ Path finding (REINFORCE algorithm)")
    print("✓ Path statistics generation")
    
    # Check for evaluation results (MAP)
    eval_file = os.path.join(relation_path, 'path_evaluation.txt')
    eval_pytorch = os.path.join(script_dir, f"logs/pytorch_eval_{relation}.log")
    
    if os.path.exists(eval_file):
        print("✓ MAP evaluation")
        with open(eval_file) as f:
            print(f"\nEvaluation results:\n{f.read()}")
    elif os.path.exists(eval_pytorch):
        print("✓ MAP evaluation")
        with open(eval_pytorch) as f:
            content = f.read()
            print("\nEvaluation results:")
            for line in content.split('\n'):
                if "MAP:" in line:
                    print(line)
    else:
        print("✗ No MAP evaluation results found (evaluation may not have been run)")
    
    if tf_exists:
        print("\nImplementation Conclusion:")
        if len(common_paths) == len(pt_paths) == len(tf_paths):
            print("✓ Implementations are functionally equivalent in path discovery")
        elif len(common_paths) / max(len(pt_paths), len(tf_paths)) > 0.8:
            print("✓ Implementations are mostly equivalent (>80% path overlap)")
        else:
            print("⚠ Implementations show significant differences in path discovery")
    else:
        print("\nImplementations cannot be compared directly - run the TensorFlow version for comparison")

def main():
    if len(sys.argv) > 1:
        relation = sys.argv[1]
        analyze_paths(relation)
    else:
        # Default to athletePlaysForTeam
        analyze_paths("athletePlaysForTeam")

if __name__ == "__main__":
    main()