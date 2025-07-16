# MINERVA PyTorch Implementation

This is a PyTorch implementation of MINERVA (Meandering In Networks of Entities to Reach Verisimilar Answers), a reinforcement learning agent for knowledge graph reasoning.

## Overview

The implementation includes:
- **Agent**: LSTM-based policy network with entity and relation embeddings
- **Environment**: Knowledge graph navigation environment with reward signals
- **Trainer**: REINFORCE algorithm with baseline for variance reduction
- **Data Loading**: Utilities for loading knowledge graphs and creating batches

## Key Differences from TensorFlow Version

1. **Modern PyTorch API**: Uses PyTorch's eager execution and autograd
2. **Simplified Architecture**: Cleaner separation of concerns
3. **Device Agnostic**: Automatically detects and uses GPU if available
4. **Better Logging**: Integrated logging throughout

## File Structure

```
minerva/src/
├── model/
│   ├── agent.py        # Policy network implementation
│   ├── environment.py  # KG navigation environment
│   ├── trainer.py      # Training loop and evaluation
│   └── baseline.py     # Baseline for variance reduction
├── data/
│   ├── grapher.py      # Knowledge graph utilities
│   └── feed_data.py    # Data loading and batching
├── options.py          # Configuration and arguments
└── train.py           # Main training script
```

## Usage

### Training

```bash
python train.py \
    --data_input_dir path/to/dataset \
    --vocab_dir path/to/vocab \
    --batch_size 128 \
    --num_rollouts 20 \
    --path_length 3 \
    --learning_rate 0.001
```

### Testing

```bash
python train.py \
    --load_model 1 \
    --model_load_dir path/to/saved/model.pt \
    --data_input_dir path/to/dataset \
    --vocab_dir path/to/vocab
```

## Key Parameters

- `--path_length`: Number of reasoning steps (default: 3)
- `--num_rollouts`: Number of rollouts during training (default: 20)
- `--test_rollouts`: Number of rollouts during testing (default: 100)
- `--hidden_size`: LSTM hidden size (default: 50)
- `--embedding_size`: Embedding dimension (default: 50)
- `--use_entity_embeddings`: Whether to use entity embeddings (0/1)

## Model Architecture

1. **Embeddings**: Separate embeddings for relations and entities
2. **LSTM Policy**: Multi-layer LSTM that maintains state across steps
3. **Action Scoring**: MLP that scores candidate actions based on state and query
4. **Sampling**: Actions sampled from softmax distribution during training

## Training Algorithm

- **REINFORCE** with cumulative discounted rewards
- **Baseline** for variance reduction (exponential moving average)
- **Entropy regularization** to encourage exploration
- **Gradient clipping** for stability

## Evaluation Metrics

- Hits@{1,3,5,10,20}: Percentage of queries where correct answer is in top-k
- MRR (Mean Reciprocal Rank): Average of 1/rank for correct answers