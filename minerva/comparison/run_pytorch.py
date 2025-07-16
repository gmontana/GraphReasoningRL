#!/usr/bin/env python
"""
Run the PyTorch MINERVA implementation
"""
import os
import sys
import json
import subprocess
import argparse

# Add the src code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

def run_pytorch_minerva(dataset='countries_S1', num_iterations=100, load_model=False):
    """Run PyTorch MINERVA on specified dataset"""
    
    # Base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, '../src')
    original_dir = os.path.join(base_dir, '../original')
    
    # Dataset paths (use same data as TF version)
    data_dir = os.path.join(original_dir, f'datasets/data_preprocessed/{dataset}')
    vocab_dir = os.path.join(data_dir, 'vocab')
    
    # Output directory
    output_dir = os.path.join(base_dir, f'outputs_pytorch/{dataset}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        'python', os.path.join(src_dir, 'train.py'),
        '--data_input_dir', data_dir,
        '--vocab_dir', vocab_dir,
        '--base_output_dir', output_dir,
        '--total_iterations', str(num_iterations),
        '--eval_every', str(max(10, num_iterations // 10)),
        '--path_length', '3',
        '--batch_size', '128',
        '--num_rollouts', '20',
        '--test_rollouts', '100',
        '--learning_rate', '0.001',
        '--beta', '0.01',
        '--gamma', '1.0',
        '--use_entity_embeddings', '0',
        '--train_entity_embeddings', '0',
        '--train_relation_embeddings', '1'
    ]
    
    if load_model:
        model_path = os.path.join(output_dir, 'model/model.pt')
        if os.path.exists(model_path):
            cmd.extend(['--load_model', '1', '--model_load_dir', model_path])
        else:
            print(f"Warning: No saved model found at {model_path}")
    
    print(f"Running PyTorch MINERVA on {dataset}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               universal_newlines=True, bufsize=1)
    
    # Capture output
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    
    process.wait()
    
    # Parse results
    results = {
        'dataset': dataset,
        'framework': 'pytorch',
        'iterations': num_iterations,
        'metrics': {}
    }
    
    # Extract final metrics from output
    metrics_found = set()
    for line in reversed(output_lines):
        if 'Final test results:' in line:
            # Start looking for metrics after this line
            continue
        if 'hits@1:' in line and 'hits@1' not in metrics_found:
            results['metrics']['hits@1'] = float(line.split(':')[-1].strip())
            metrics_found.add('hits@1')
        elif 'hits@3:' in line and 'hits@3' not in metrics_found:
            results['metrics']['hits@3'] = float(line.split(':')[-1].strip())
            metrics_found.add('hits@3')
        elif 'hits@5:' in line and 'hits@5' not in metrics_found:
            results['metrics']['hits@5'] = float(line.split(':')[-1].strip())
            metrics_found.add('hits@5')
        elif 'hits@10:' in line and 'hits@10' not in metrics_found:
            results['metrics']['hits@10'] = float(line.split(':')[-1].strip())
            metrics_found.add('hits@10')
        elif 'mrr:' in line and 'mrr' not in metrics_found:
            results['metrics']['mrr'] = float(line.split(':')[-1].strip())
            metrics_found.add('mrr')
        
        if len(metrics_found) == 5:
            break  # Found all metrics
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run PyTorch MINERVA')
    parser.add_argument('--dataset', default='countries_S1', 
                        help='Dataset to use (e.g., countries_S1, kinship)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--load_model', action='store_true',
                        help='Load pretrained model if available')
    
    args = parser.parse_args()
    
    # Check if PyTorch is installed
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("MPS device: Apple Silicon GPU")
    except ImportError:
        print("PyTorch not installed. Please install with:")
        print("pip install -r requirements_pytorch.txt")
        sys.exit(1)
    
    results = run_pytorch_minerva(args.dataset, args.iterations, args.load_model)
    print("\nFinal metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()