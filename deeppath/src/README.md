# DeepPath PyTorch Implementation

This directory contains the PyTorch-specific implementation details for DeepPath. For algorithm explanation and project overview, see the [main README](../README.md).

## PyTorch-Specific Features

- **Multi-Device Support**: Automatic detection and support for CUDA, MPS (Apple Silicon), and CPU
- **Modern PyTorch Patterns**: nn.Module architecture, eager execution, native optimizers
- **Enhanced Training**: Improved stability with gradient clipping and proper initialization
- **Flexible Architecture**: Modular design for easy experimentation

## Quick Start

```bash
# Train supervised policy
python main.py athletePlaysForTeam --mode train_sl

# Train with reinforcement learning
python main.py athletePlaysForTeam --mode train_rl

# Test the model
python main.py athletePlaysForTeam --mode test

# Run full pipeline (supervised + RL + test)
python main.py athletePlaysForTeam
```

## Model Components

### Core Modules

- **`agents.py`**: Agent implementations
  - `PolicyAgent`: REINFORCE-based RL agent
  - `SupervisedPolicyAgent`: Supervised learning from teacher paths
  - `train_reinforce()`: REINFORCE training loop

- **`models.py`**: Neural network architectures
  - `PolicyNetwork`: LSTM-based policy network
  - `ValueNetwork`: State value estimation
  - `QNetwork`: Action-value estimation

- **`environment.py`**: Knowledge graph environment
  - State representation and transitions
  - Action space management
  - Reward calculation

- **`search.py`**: Path finding algorithms
  - BFS teacher for supervised learning
  - Path diversity calculation
  - Knowledge base utilities

- **`evaluate.py`**: Evaluation logic
  - Path ranking and selection
  - Mean Average Precision (MAP) calculation
  - Test set evaluation

### Training Process

#### 1. Supervised Learning Phase
```python
# Teacher finds paths using BFS
teacher_paths = teacher(e1, e2, num_paths)

# Agent learns to imitate teacher
supervised_agent.learn(teacher_paths)
```

#### 2. Reinforcement Learning Phase
```python
# Agent explores and learns from rewards
train_reinforce(
    agent,
    environment,
    episodes=num_episodes,
    gamma=0.99  # discount factor
)
```

#### 3. Path Evaluation
```python
# Rank discovered paths by efficiency
path_scores = rank_paths(paths)

# Evaluate on test set
map_score = evaluate_paths(test_pairs, ranked_paths)
```

## Configuration Options

### Training Parameters
```python
# Supervised learning
--episodes_sl      # Number of supervised episodes (default: 1000)
--learning_rate_sl # Learning rate for supervised (default: 0.001)

# Reinforcement learning  
--episodes_rl      # Number of RL episodes (default: 500)
--learning_rate_rl # Learning rate for RL (default: 0.0001)
--gamma           # Discount factor (default: 0.99)

# Model architecture
--hidden_size     # LSTM hidden size (default: 512)
--embedding_size  # Entity/relation embeddings (default: 200)
```

### Device Support

The implementation automatically detects the best available device:

```python
# In order of preference:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU

# Check detected device:
python -c "from src.utils import select_device; print(select_device())"
```

## Implementation Notes

### Key Differences from TensorFlow

1. **State Representation**: Uses PyTorch tensors with automatic differentiation
2. **Training Loop**: Native PyTorch optimizers and loss computation
3. **Model Architecture**: nn.Module-based with cleaner forward passes

### Performance Optimizations

1. **Batch Processing**: Efficient batched LSTM operations
2. **GPU Acceleration**: Full support for CUDA and MPS
3. **Memory Management**: Proper tensor cleanup and gradient zeroing

### Debugging Features

1. **Verbose Mode**: Detailed logging of training progress
2. **Model Checkpointing**: Save/load model states
3. **Path Visualization**: Export discovered paths for analysis

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in supervised learning
   - Decrease number of parallel episodes in RL
   - Use CPU for debugging

2. **Slow Training**
   - Ensure GPU is being used (check device selection)
   - Reduce embedding dimensions for faster computation
   - Use fewer BFS paths in supervised learning

3. **Poor Performance**
   - Increase number of training episodes
   - Tune learning rates (try grid search)
   - Check if teacher paths are diverse enough

## Contributing

When contributing, please:
1. Follow PyTorch best practices
2. Maintain compatibility with different devices
3. Add appropriate error handling
4. Update documentation for new features