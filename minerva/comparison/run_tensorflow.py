#!/usr/bin/env python
"""
Run the original TensorFlow MINERVA implementation
"""
import os
import sys
import json
import subprocess
import argparse

# Add the original code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../original/code'))

def run_tensorflow_minerva(dataset='countries_S1', num_iterations=100, load_model=False):
    """Run TensorFlow MINERVA on specified dataset"""
    
    # Base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_dir = os.path.join(base_dir, '../original')
    
    # Dataset paths
    data_dir = os.path.join(original_dir, f'datasets/data_preprocessed/{dataset}')
    vocab_dir = os.path.join(data_dir, 'vocab')
    
    # Output directory
    output_dir = os.path.join(base_dir, f'outputs_tf/{dataset}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        'python', os.path.join(original_dir, 'code/model/trainer.py'),
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
        model_path = os.path.join(original_dir, f'saved_models/{dataset}/model.ckpt')
        if os.path.exists(model_path + '.index'):
            cmd.extend(['--load_model', '1', '--model_load_dir', model_path])
        else:
            print(f"Warning: No saved model found at {model_path}")
    
    # Set environment to suppress TF warnings
    env = os.environ.copy()
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print(f"Running TensorFlow MINERVA on {dataset}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
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
        'framework': 'tensorflow',
        'iterations': num_iterations,
        'metrics': {}
    }
    
    # Extract final metrics from output
    for line in reversed(output_lines):
        if 'Hits@1:' in line:
            results['metrics']['hits@1'] = float(line.split(':')[1].strip())
        elif 'Hits@3:' in line:
            results['metrics']['hits@3'] = float(line.split(':')[1].strip())
        elif 'Hits@5:' in line:
            results['metrics']['hits@5'] = float(line.split(':')[1].strip())
        elif 'Hits@10:' in line:
            results['metrics']['hits@10'] = float(line.split(':')[1].strip())
        elif 'auc:' in line or 'MRR:' in line:
            results['metrics']['mrr'] = float(line.split(':')[1].strip())
            break  # Found all metrics
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run TensorFlow MINERVA')
    parser.add_argument('--dataset', default='countries_S1', 
                        help='Dataset to use (e.g., countries_S1, kinship)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--load_model', action='store_true',
                        help='Load pretrained model if available')
    
    args = parser.parse_args()
    
    # Check if TensorFlow is installed
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("TensorFlow not installed. Please install with:")
        print("pip install -r requirements_tf.txt")
        sys.exit(1)
    
    results = run_tensorflow_minerva(args.dataset, args.iterations, args.load_model)
    print("\nFinal metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()