# MINERVA PyTorch Implementation

This is a complete PyTorch implementation of MINERVA (Meandering In Networks of Entities to Reach Verisimilar Answers), a reinforcement learning agent for knowledge graph reasoning.

## Features

### ✅ Complete Feature Parity with Original

This implementation includes all features from the original TensorFlow version:

1. **Core Algorithm**
   - REINFORCE with baseline
   - Cumulative discounted rewards
   - Entropy regularization with decaying beta
   - Reactive baseline with exponential moving average

2. **Model Architecture**
   - LSTM policy network with configurable layers
   - Entity and relation embeddings
   - Optional entity embedding usage
   - Pretrained embedding support

3. **Evaluation Methods**
   - Beam search for inference
   - Max and sum pooling strategies
   - Hits@1/3/5/10/20 metrics
   - Mean Reciprocal Rank (MRR)
   - NELL evaluation system

4. **Data Processing**
   - Vocabulary creation scripts
   - NELL-specific preprocessing
   - Support for all 24 original datasets
   - JSON configuration files

### 🚀 Enhanced Features

1. **Multi-Device Support**
   - Automatic device detection
   - CUDA support for NVIDIA GPUs
   - MPS support for Apple Silicon (M1/M2/M3/M4)
   - CPU fallback

2. **Modern PyTorch Implementation**
   - Clean module structure
   - Type hints where applicable
   - Comprehensive error handling
   - Detailed logging

3. **Additional Features**
   - L2 regularization support
   - JSON configuration loading
   - Command-line override for configs
   - Progress bars with tqdm

## Installation

```bash
# Basic requirements
pip install torch numpy tqdm

# For CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For development
pip install pytest black isort
```

## Quick Start

### Using JSON Configuration (Recommended)

```bash
# Train using a predefined configuration
python train.py ../configs/countries_s1.json

# Override specific parameters
python train.py ../configs/countries_s1.json --total_iterations 2000 --batch_size 512

# Specify device
python train.py ../configs/kinship.json --device cuda
python train.py ../configs/kinship.json --device mps   # Apple Silicon
```

### Manual Configuration

```bash
python train.py \
    --data_input_dir ../original/datasets/data_preprocessed/countries_S1 \
    --vocab_dir ../original/datasets/data_preprocessed/countries_S1/vocab \
    --path_length 2 \
    --hidden_size 25 \
    --embedding_size 25 \
    --batch_size 256 \
    --total_iterations 1000 \
    --eval_every 100
```

## Model Components

### Core Modules

- **`model/agent.py`**: LSTM policy network
  - Configurable LSTM layers
  - Entity and relation embeddings
  - Action selection with masking

- **`model/trainer.py`**: Training loop
  - REINFORCE implementation
  - Beam search evaluation
  - Model checkpointing

- **`model/environment.py`**: KG navigation
  - Episode management
  - State transitions
  - Reward calculation

- **`model/baseline.py`**: Variance reduction
  - Reactive baseline
  - Exponential moving average

- **`model/nell_eval.py`**: NELL evaluation
  - MAP calculation
  - Special handling for NELL format

### Data Processing

- **`data/grapher.py`**: Knowledge graph representation
  - Efficient array-based storage
  - Action filtering
  - NO_OP self-loops

- **`data/feed_data.py`**: Batch generation
  - Train/dev/test splits
  - Rollout expansion
  - Answer tracking

- **`data/create_vocab.py`**: Vocabulary creation
  - Generic vocabulary builder
  - Special token handling

- **`data/preprocess_nell.py`**: NELL preprocessing
  - NELL-specific format handling
  - Test pairs processing

## Configuration Options

### Essential Parameters

```python
# Model architecture
--path_length        # Number of reasoning steps (default: 3)
--hidden_size        # LSTM hidden size (default: 50)
--embedding_size     # Embedding dimensions (default: 50)
--LSTM_layers        # Number of LSTM layers (default: 1)

# Training
--batch_size         # Training batch size (default: 128)
--num_rollouts       # Rollouts during training (default: 20)
--test_rollouts      # Rollouts during testing (default: 100)
--total_iterations   # Total training steps (default: 2000)
--eval_every         # Evaluation frequency (default: 100)

# Optimization
--learning_rate      # Learning rate (default: 0.001)
--grad_clip_norm     # Gradient clipping (default: 5)
--l2_reg_const       # L2 regularization (default: 0.01)
--beta               # Entropy coefficient (default: 0.01)
--gamma              # Discount factor (default: 1.0)
--Lambda             # Baseline decay (default: 0.0)
```

### Advanced Options

```python
# Embeddings
--use_entity_embeddings       # Use entity embeddings (0/1)
--train_entity_embeddings     # Train entity embeddings (0/1)
--train_relation_embeddings   # Train relation embeddings (0/1)
--pretrained_embeddings_action # Path to pretrained relations
--pretrained_embeddings_entity # Path to pretrained entities

# Evaluation
--pool                        # Pooling method: 'max' or 'sum'
--nell_evaluation             # Enable NELL evaluation (0/1)

# Device
--device                      # Device: 'cuda', 'mps', 'cpu', None
```

## Datasets

### Supported Datasets

All 24 original datasets are supported with JSON configurations:

**Knowledge Graphs:**
- FB15K-237
- WN18RR
- NELL-995
- UMLS
- Kinship

**Countries (Synthetic):**
- Countries S1/S2/S3

**Grid Worlds:**
- Grid 4x4, 6x6, 8x8, 10x10

**NELL Relations:**
- agentbelongstoorganization
- athleteplaysforteam
- athleteplaysinleague
- athleteplayssport
- organizationheadquarteredincity
- personborninlocation
- teamplaysinleague
- And more...

### Data Format

Datasets should contain:
```
dataset_dir/
├── train.txt          # Training triples
├── dev.txt            # Development triples
├── test.txt           # Test triples
├── graph.txt          # Full graph for navigation
└── vocab/
    ├── entity_vocab.json
    └── relation_vocab.json
```

### Creating New Datasets

```bash
# Generic vocabulary creation
python data/create_vocab.py \
    --data_input_dir /path/to/dataset \
    --vocab_dir /path/to/dataset/vocab

# NELL-specific preprocessing
python data/preprocess_nell.py \
    --task_dir /path/to/nell/task \
    --task_name athleteplayssport \
    --output_dir /path/to/output
```

## Training Process

### What Happens During Training

1. **Initialization**
   - Load vocabularies and build knowledge graph
   - Initialize embeddings (Xavier/zeros)
   - Setup LSTM policy network

2. **Training Loop**
   - Sample batch of (entity, relation, ?) queries
   - Execute rollouts using current policy
   - Calculate rewards (binary: correct/incorrect)
   - Compute cumulative discounted rewards
   - Update policy using REINFORCE
   - Update baseline for variance reduction

3. **Evaluation**
   - Run beam search on dev set
   - Calculate Hits@k and MRR
   - Save model if Hits@10 improves

4. **Final Testing**
   - Load best model
   - Evaluate on test set
   - Run NELL evaluation if enabled

### Monitoring Training

Training produces detailed logs:
```
Batch 100: hits=19.5000, avg_reward=0.0977, num_correct=7, accuracy=0.2188, loss=2.3456
Evaluating on dev set...
Hits@1: 0.4583
Hits@3: 0.9167
Hits@5: 0.9583
Hits@10: 0.9583
MRR: 0.6806
```

## Device Support

### Automatic Detection

The implementation automatically detects the best available device:

```python
# In order of preference:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU

# Check detected device:
python train.py --help
# Shows: "Detected device: cuda/mps/cpu"
```

### Manual Selection

```bash
# Force specific device
python train.py ../configs/fb15k-237.json --device cuda
python train.py ../configs/fb15k-237.json --device mps
python train.py ../configs/fb15k-237.json --device cpu
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch_size
   - Reduce num_rollouts
   - Use CPU for debugging

2. **Slow Training**
   - Ensure GPU is being used
   - Check batch_size is appropriate
   - Reduce test_rollouts for faster evaluation

3. **Poor Performance**
   - Increase total_iterations
   - Tune learning_rate
   - Adjust beta for exploration

4. **NELL Evaluation Not Working**
   - Ensure sort_test.pairs exists
   - Set nell_evaluation=1
   - Check path tracking is enabled

## Implementation Notes

### Key Differences from TensorFlow

1. **Framework APIs**
   - PyTorch nn.Module vs TF layers
   - Eager execution vs graph mode
   - Different optimizer implementations

2. **Numerical Precision**
   - Minor differences expected
   - Same algorithms, different backends
   - Results should be within 1-5%

### Verified Components

All critical components have been verified against the original:
- ✅ Reward normalization formula
- ✅ Entropy calculation
- ✅ Cumulative reward discounting
- ✅ Beta decay schedule
- ✅ Action masking value (-99999.0)
- ✅ Log-sum-exp for sum pooling

## Contributing

When contributing, please:
1. Maintain compatibility with original
2. Add tests for new features
3. Update documentation
4. Follow existing code style

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{minerva,
  title = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and Vilnis, Luke and Durugkar, Ishan and Krishnamurthy, Akshay and Smola, Alex and McCallum, Andrew},
  booktitle = {ICLR},
  year = {2018}
}
```