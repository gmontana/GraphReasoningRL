#!/usr/bin/env python3
"""
Generate JSON configuration files from the original shell script configurations.

This script converts all the original .sh config files to JSON format
that can be used by the PyTorch implementation.
"""

import json
import os
import re
from pathlib import Path


def parse_shell_config(filepath):
    """Parse a shell script configuration file and extract variables."""
    config = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                # Extract variable assignment
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                # Convert to appropriate type
                if value.lower() in ['true', 'false']:
                    config[key] = value.lower() == 'true'
                elif value.isdigit():
                    config[key] = int(value)
                elif value.replace('.', '').isdigit():
                    config[key] = float(value)
                elif value.lower() == 'null':
                    config[key] = None
                else:
                    config[key] = value
    
    return config


def convert_config_for_pytorch(config):
    """Convert shell config to PyTorch trainer format."""
    # Default values for all configurations
    pytorch_config = {
        # Data parameters
        "data_input_dir": config.get("data_input_dir", ""),
        "input_file": "train.txt",
        "create_vocab": 0,
        "vocab_dir": config.get("vocab_dir", ""),
        "max_num_actions": 200,
        
        # Model parameters
        "path_length": config.get("path_length", 3),
        "hidden_size": config.get("hidden_size", 50),
        "embedding_size": config.get("embedding_size", 50),
        "LSTM_layers": 1,
        "use_entity_embeddings": config.get("use_entity_embeddings", 0),
        "train_entity_embeddings": config.get("train_entity_embeddings", 0),
        "train_relation_embeddings": config.get("train_relation_embeddings", 1),
        
        # Training parameters
        "batch_size": config.get("batch_size", 128),
        "num_rollouts": 20,
        "test_rollouts": 100,
        "learning_rate": 1e-3,
        "grad_clip_norm": 5,
        "l2_reg_const": 1e-2,
        "beta": config.get("beta", 1e-2),
        "gamma": 1.0,
        "Lambda": config.get("Lambda", 0.0),
        
        # Reward parameters
        "positive_reward": 1.0,
        "negative_reward": 0.0,
        
        # Training control
        "total_iterations": config.get("total_iterations", 2000),
        "eval_every": 100,
        "pool": "max",
        
        # Model saving/loading
        "base_output_dir": config.get("base_output_dir", "./outputs").rstrip('/'),
        "load_model": config.get("load_model", 0),
        "model_load_dir": config.get("model_load_dir", ""),
        
        # Logging
        "log_dir": "./logs/",
        "log_file_name": "train.log",
        "output_file": "",
        
        # NELL evaluation
        "nell_evaluation": config.get("nell_evaluation", 0),
        
        # Pretrained embeddings
        "pretrained_embeddings_action": "",
        "pretrained_embeddings_entity": "",
        
        # Device
        "device": None
    }
    
    # Handle null model_load_dir
    if pytorch_config["model_load_dir"] == "null":
        pytorch_config["model_load_dir"] = ""
    
    return pytorch_config


def main():
    """Generate all configuration files."""
    original_configs_dir = "../original/configs"
    output_dir = "."
    
    # Check if original configs directory exists
    if not os.path.exists(original_configs_dir):
        print(f"Error: Original configs directory not found: {original_configs_dir}")
        return
    
    # Process all .sh files
    config_files = []
    for filename in os.listdir(original_configs_dir):
        if filename.endswith('.sh'):
            config_files.append(filename)
    
    config_files.sort()
    print(f"Found {len(config_files)} configuration files to convert")
    
    for filename in config_files:
        shell_path = os.path.join(original_configs_dir, filename)
        
        # Parse the shell config
        try:
            shell_config = parse_shell_config(shell_path)
            pytorch_config = convert_config_for_pytorch(shell_config)
            
            # Create output filename
            dataset_name = filename.replace('.sh', '')
            json_filename = f"{dataset_name}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            # Write JSON config
            with open(json_path, 'w') as f:
                json.dump(pytorch_config, f, indent=2)
            
            print(f"✓ Converted {filename} -> {json_filename}")
            
        except Exception as e:
            print(f"✗ Error converting {filename}: {e}")
    
    # Create a README file
    readme_content = """# MINERVA Configuration Files

This directory contains JSON configuration files for all supported datasets in the MINERVA PyTorch implementation.

## Usage

To use a configuration file with the PyTorch implementation:

```bash
cd minerva/src
python train.py @../configs/countries_s1.json
```

Or load specific parameters:
```bash
python train.py --data_input_dir ../original/datasets/data_preprocessed/countries_S1 \\
                --vocab_dir ../original/datasets/data_preprocessed/countries_S1/vocab \\
                --path_length 2 --hidden_size 25 --embedding_size 25 \\
                --batch_size 256 --beta 0.05 --Lambda 0.05 \\
                --use_entity_embeddings 1 --train_entity_embeddings 1 \\
                --total_iterations 1000
```

## Configuration Files

The following datasets are supported:

### Knowledge Graph Completion
- `fb15k-237.json` - FB15k-237 dataset
- `WN18RR.json` - WordNet 18 RR dataset
- `umls.json` - UMLS dataset
- `kinship.json` - Kinship dataset

### Countries Dataset Variants
- `countries_s1.json` - Countries S1
- `countries_s2.json` - Countries S2  
- `countries_s3.json` - Countries S3

### Grid World Environments
- `grid_4.json` - 4x4 grid world
- `grid_6.json` - 6x6 grid world
- `grid_8.json` - 8x8 grid world
- `grid_10.json` - 10x10 grid world

### NELL Dataset Relations
- `agentbelongstoorganization.json`
- `athletehomestadium.json`
- `athleteplaysforteam.json`
- `athleteplaysinleague.json`
- `athleteplayssport.json`
- `organizationheadquarteredincity.json`
- `organizationhiredperson.json`
- `personborninlocation.json`
- `personleadsorganization.json`
- `teamplaysinleague.json`
- `teamplayssport.json`
- `worksfor.json`
- `nell.json` - General NELL configuration

## Configuration Parameters

Each configuration file contains the following parameters:

### Data Parameters
- `data_input_dir`: Path to preprocessed dataset
- `vocab_dir`: Path to vocabulary files
- `input_file`: Training file name (default: "train.txt")
- `max_num_actions`: Maximum actions per state (default: 200)

### Model Parameters  
- `path_length`: Reasoning path length
- `hidden_size`: LSTM hidden size
- `embedding_size`: Entity/relation embedding size
- `use_entity_embeddings`: Whether to use entity embeddings
- `train_entity_embeddings`: Whether to train entity embeddings
- `train_relation_embeddings`: Whether to train relation embeddings

### Training Parameters
- `batch_size`: Training batch size
- `total_iterations`: Total training iterations
- `learning_rate`: Learning rate (default: 1e-3)
- `beta`: Entropy regularization weight
- `Lambda`: Baseline learning rate
- `gamma`: Discount factor (default: 1.0)

### Evaluation
- `pool`: Pooling method ("max" or "sum")
- `nell_evaluation`: Enable NELL evaluation (0/1)
- `eval_every`: Evaluation frequency

## Notes

- All paths in the configuration files are relative to the MINERVA root directory
- NELL evaluation is enabled for NELL-specific dataset configurations
- Grid world datasets use different hyperparameters optimized for those environments
- The original TensorFlow model paths have been cleared in the `model_load_dir` field
"""
    
    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write(readme_content)
    
    print(f"\n✓ Generated README.md")
    print(f"\nConversion complete! {len(config_files)} configuration files created.")


if __name__ == '__main__':
    main()