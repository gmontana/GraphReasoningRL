# MINERVA Configuration Files

This directory contains JSON configuration files for all supported datasets in the MINERVA PyTorch implementation.

## Usage

To use a configuration file with the PyTorch implementation:

```bash
cd minerva/src
python train.py ../configs/countries_s1.json
```

Or load specific parameters:
```bash
python train.py --data_input_dir ../original/datasets/data_preprocessed/countries_S1 \
                --vocab_dir ../original/datasets/data_preprocessed/countries_S1/vocab \
                --path_length 2 --hidden_size 25 --embedding_size 25 \
                --batch_size 256 --beta 0.05 --Lambda 0.05 \
                --use_entity_embeddings 1 --train_entity_embeddings 1 \
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
