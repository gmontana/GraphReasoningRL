#!/usr/bin/env python
"""
Script to run the evaluation module on a trained DeepPath model.
"""

import sys
import os
import torch

# Check for CUDA or MPS availability and set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using CPU device")

print(f"Using device: {device}")

from src.evaluate import evaluate_logic

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_evaluation.py <relation>")
        print("Example: python run_evaluation.py athletePlaysForTeam")
        return
    
    relation = sys.argv[1]
    data_path = './NELL-995/tasks/'
    
    # Verify relation directory exists
    relation_path = os.path.join(data_path, relation)
    if not os.path.exists(relation_path):
        print(f"Error: Relation directory not found: {relation_path}")
        return
    
    # Check for necessary files
    graph_path = os.path.join(relation_path, 'graph.txt')
    path_to_use = os.path.join(relation_path, 'path_to_use.txt')
    path_stats = os.path.join(relation_path, 'path_stats.txt')
    
    if not all(os.path.exists(p) for p in [graph_path, path_to_use, path_stats]):
        print("Error: Required files not found. Please run the model first:")
        print("  python main.py athletePlaysForTeam")
        return
    
    print(f"Running evaluation for relation: {relation}")
    print("This will train a prediction model on the discovered paths")
    
    # Run evaluation
    mean_ap = evaluate_logic(relation, data_path)
    
    print(f"\nEvaluation complete for {relation}")
    if mean_ap is not None:
        print(f"Mean Average Precision (MAP): {mean_ap:.4f}")
    else:
        print("Unable to calculate Mean Average Precision.")

if __name__ == "__main__":
    main()