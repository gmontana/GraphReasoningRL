import argparse
import uuid
import os
import json
from pprint import pprint


def load_config_from_json(json_path):
    """Load configuration from JSON file and convert to command line format."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Convert JSON config to command line arguments
    args = []
    for key, value in config.items():
        if value is not None and value != "":
            args.append(f"--{key}")
            args.append(str(value))
    
    return args


def read_options():
    parser = argparse.ArgumentParser(description='MINERVA: PyTorch Implementation',
                                   fromfile_prefix_chars='@')
    
    # Data parameters
    parser.add_argument("--data_input_dir", default="", type=str, help="Input data directory")
    parser.add_argument("--input_file", default="train.txt", type=str, help="Input training file name")
    parser.add_argument("--create_vocab", default=0, type=int, help="Create vocabulary files (0/1)")
    parser.add_argument("--vocab_dir", default="", type=str, help="Vocabulary directory")
    parser.add_argument("--max_num_actions", default=200, type=int, help="Maximum number of actions per state")
    
    # Model parameters
    parser.add_argument("--path_length", default=3, type=int, help="Path length for reasoning")
    parser.add_argument("--hidden_size", default=50, type=int, help="Hidden size for LSTM")
    parser.add_argument("--embedding_size", default=50, type=int, help="Embedding size")
    parser.add_argument("--LSTM_layers", default=1, type=int, help="Number of LSTM layers")
    parser.add_argument("--use_entity_embeddings", default=0, type=int, help="Use entity embeddings (0/1)")
    parser.add_argument("--train_entity_embeddings", default=0, type=int, help="Train entity embeddings (0/1)")
    parser.add_argument("--train_relation_embeddings", default=1, type=int, help="Train relation embeddings (0/1)")
    
    # Training parameters
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--num_rollouts", default=20, type=int, help="Number of rollouts during training")
    parser.add_argument("--test_rollouts", default=100, type=int, help="Number of rollouts during testing")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--grad_clip_norm", default=5, type=int, help="Gradient clipping norm")
    parser.add_argument("--l2_reg_const", default=1e-2, type=float, help="L2 regularization constant")
    parser.add_argument("--beta", default=1e-2, type=float, help="Entropy regularization weight")
    parser.add_argument("--gamma", default=1, type=float, help="Discount factor")
    parser.add_argument("--Lambda", default=0.0, type=float, help="Baseline learning rate")
    
    # Reward parameters
    parser.add_argument("--positive_reward", default=1.0, type=float, help="Reward for correct answer")
    parser.add_argument("--negative_reward", default=0, type=float, help="Reward for incorrect answer")
    
    # Training control
    parser.add_argument("--total_iterations", default=2000, type=int, help="Total training iterations")
    parser.add_argument("--eval_every", default=100, type=int, help="Evaluate every N iterations")
    parser.add_argument("--pool", default="max", type=str, help="Pooling method for evaluation")
    
    # Model saving/loading
    parser.add_argument("--model_dir", default='', type=str, help="Model directory")
    parser.add_argument("--base_output_dir", default='./outputs', type=str, help="Base output directory")
    parser.add_argument("--load_model", default=0, type=int, help="Load pretrained model (0/1)")
    parser.add_argument("--model_load_dir", default="", type=str, help="Path to load model from")
    
    # Logging
    parser.add_argument("--log_dir", default="./logs/", type=str, help="Log directory")
    parser.add_argument("--log_file_name", default="train.log", type=str, help="Log file name")
    parser.add_argument("--output_file", default="", type=str, help="Output file for results")
    
    # NELL evaluation
    parser.add_argument("--nell_evaluation", default=0, type=int, help="Perform NELL evaluation (0/1)")
    
    # Pretrained embeddings
    parser.add_argument("--pretrained_embeddings_action", default="", type=str, help="Path to pretrained action/relation embeddings")
    parser.add_argument("--pretrained_embeddings_entity", default="", type=str, help="Path to pretrained entity embeddings")
    
    # Device
    parser.add_argument("--device", default=None, type=str, 
                        help="Device to use: 'cuda', 'mps', 'cpu', or None for auto-detection")
    
    # Check if we're loading from JSON config file
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        # Load from JSON file
        json_config_path = sys.argv[1]
        if not os.path.exists(json_config_path):
            raise FileNotFoundError(f"Configuration file not found: {json_config_path}")
        
        with open(json_config_path, 'r') as f:
            parsed = json.load(f)
        
        # Override with any additional command line arguments
        if len(sys.argv) > 2:
            additional_args = parser.parse_args(sys.argv[2:])
            parsed.update(vars(additional_args))
    else:
        # Parse arguments normally
        args = parser.parse_args()
        parsed = vars(args)
    
    # Convert boolean flags
    parsed['use_entity_embeddings'] = bool(parsed['use_entity_embeddings'])
    parsed['train_entity_embeddings'] = bool(parsed['train_entity_embeddings'])
    parsed['train_relation_embeddings'] = bool(parsed['train_relation_embeddings'])
    parsed['load_model'] = bool(parsed['load_model'])
    
    # Create input_files list for compatibility
    if parsed['data_input_dir'] and parsed['input_file']:
        parsed['input_files'] = [os.path.join(parsed['data_input_dir'], parsed['input_file'])]
    else:
        parsed['input_files'] = []
    
    # Load vocabularies
    if parsed['vocab_dir']:
        parsed['entity_vocab'] = json.load(open(os.path.join(parsed['vocab_dir'], 'entity_vocab.json')))
        parsed['relation_vocab'] = json.load(open(os.path.join(parsed['vocab_dir'], 'relation_vocab.json')))
    
    # Create output directory
    if not parsed['load_model']:
        run_id = str(uuid.uuid4())[:8]
        parsed['output_dir'] = os.path.join(
            parsed['base_output_dir'],
            f"{run_id}_pl{parsed['path_length']}_b{parsed['beta']}_tr{parsed['test_rollouts']}_l{parsed['Lambda']}"
        )
        parsed['model_dir'] = os.path.join(parsed['output_dir'], 'model')
        parsed['log_file_name'] = os.path.join(parsed['output_dir'], 'train.log')
        parsed['path_logger_file'] = parsed['output_dir']  # For compatibility with original
        
        # Create directories
        os.makedirs(parsed['output_dir'], exist_ok=True)
        os.makedirs(parsed['model_dir'], exist_ok=True)
        
        # Save config
        with open(os.path.join(parsed['output_dir'], 'config.json'), 'w') as f:
            json.dump(parsed, f, indent=2, default=str)
    
    # Print configuration
    print('Configuration:')
    max_len = max(len(k) for k in parsed.keys())
    for k, v in sorted(parsed.items()):
        print(f'  {k:<{max_len}} : {v}')
    
    return parsed