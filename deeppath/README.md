# DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning

This repository contains both the original TensorFlow implementation and our new PyTorch implementation of DeepPath, a reinforcement learning method for reasoning over knowledge graphs. DeepPath finds reasoning paths between entities in knowledge graphs by training an agent to navigate from source to target entities.

## How DeepPath Works

### 1. Knowledge Graph Query Answering

DeepPath addresses the problem of finding missing links in knowledge graphs. Given a query like (LeBron James, playsForTeam, ?), the agent learns to navigate through the knowledge graph to find the answer (e.g., LA Lakers) by following a sequence of relations.

Unlike embedding-based methods, DeepPath:
- Provides interpretable reasoning paths between entities
- Learns to navigate the discrete graph structure
- Can generalize to unseen entity pairs
- Discovers multiple reasoning patterns for each relation

### 2. Agent-Environment System

- **State**: Current entity and target entity embeddings
- **Actions**: Selecting which outgoing relation to follow
- **Environment**: Knowledge graph with direct links removed during training
- **Reward**: Binary reward for reaching the correct target entity

### 3. Learning Process

1. **Supervised Learning**: Agent learns from a teacher (BFS algorithm) that finds correct paths
2. **Reinforcement Learning**: Agent refines its policy using REINFORCE algorithm
3. **Path Ranking**: Discovered paths are ranked by their predictive accuracy

### 4. Key Innovations

- **Two-phase training**: Supervised pre-training followed by RL fine-tuning
- **Path diversity**: Encourages finding multiple reasoning patterns
- **Efficiency reward**: Shorter paths are preferred
- **Teacher forcing**: BFS teacher provides initial guidance

## Algorithm Deep Dive

### The Core Idea

DeepPath treats knowledge graph reasoning as a sequential decision problem. The agent learns to walk from a source entity to a target entity by selecting relations at each step.

For example, to answer "Which team does LeBron James play for?":
1. Start at entity "LeBron James"
2. Choose relation "playsPosition" → reach "Small Forward"
3. Choose relation "positionPlayedByTeam" → reach "LA Lakers"

### The DeepPath Algorithm

#### 1. Problem Formulation

**Input**: Knowledge Graph and Query (e_source, r_query, ?)
- KG: Set of triples (head, relation, tail)
- Query: Find e_target given e_source and relation type

**Output**: Reasoning paths that connect e_source to e_target

#### 2. Markov Decision Process (MDP)

- **State s_t**: Current entity e_t and target entity e_target embeddings
- **Action a_t**: Select relation r from available relations at e_t
- **Transition**: Follow relation r to reach next entity
- **Reward**: 
  - +1 if final entity equals target
  - -1/length for efficiency bonus
  - Path diversity bonus

#### 3. Two-Phase Training

**Phase 1: Supervised Learning**
```
1. Teacher (BFS) finds correct paths
2. Agent learns to imitate teacher's actions
3. Cross-entropy loss on action selection
```

**Phase 2: Reinforcement Learning**
```
1. Agent explores using learned policy
2. REINFORCE algorithm updates policy
3. Encourages diverse path discovery
```

#### 4. Path Ranking and Evaluation

After training, paths are ranked by:
- Accuracy on validation set
- Path length (shorter is better)  
- Coverage of test entity pairs

Mean Average Precision (MAP) measures final performance.

### PyTorch Implementation Details

See [src/README.md](src/README.md) for PyTorch-specific implementation details, configuration options, and usage instructions.

## Directory Structure

```
deeppath/
├── README.md                   # This file
├── src/                        # PyTorch implementation
│   ├── agents.py               # RL and supervised agents
│   ├── environment.py          # KG navigation
│   ├── models.py               # Neural networks
│   ├── search.py               # BFS teacher
│   ├── evaluate.py             # Path evaluation
│   └── README.md               # PyTorch details
├── comparison/                 # TensorFlow comparison
│   ├── tensorflow/             # Original TF code
│   └── compare_implementations.py
├── NELL-995/                   # Dataset
│   └── tasks/                  # Relation-specific data
├── main.py                     # Entry point
└── requirements.txt            # Dependencies
```

## Installation

### Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/DeepPath.git
cd DeepPath

# Setup environment
./setup_conda_env.sh
conda activate deeppath_torch
```

### Using pip

```bash
# Clone repository
git clone https://github.com/yourusername/DeepPath.git
cd DeepPath

# Install dependencies
pip install -r requirements.txt
```

## Dataset

DeepPath uses the NELL-995 dataset. Download it from:
- [KB-Reasoning-Data](https://github.com/wenhuchen/KB-Reasoning-Data)

```bash
# Download and extract
git clone https://github.com/wenhuchen/KB-Reasoning-Data.git
cp -r KB-Reasoning-Data/NELL-995 ./
```

## Usage

### PyTorch Implementation

```bash
# Full pipeline (supervised + RL + evaluation)
python main.py athletePlaysForTeam

# Individual phases
python main.py athletePlaysForTeam --mode train_sl  # Supervised only
python main.py athletePlaysForTeam --mode train_rl  # RL only
python main.py athletePlaysForTeam --mode test      # Evaluation only
```

See [src/README.md](src/README.md) for detailed PyTorch usage and configuration.

### Comparing Implementations

```bash
cd comparison/

# Compare PyTorch vs TensorFlow
python compare_implementations.py athletePlaysForTeam

# Run full benchmark
./run_complete_benchmark.sh athletePlaysForTeam
```

## Implementation Features

### PyTorch Implementation:
- Multi-device support (CUDA, MPS, CPU)
- Modern PyTorch patterns
- Enhanced stability and logging
- Verified parity with original

### TensorFlow Implementation:
- Original implementation from paper
- TensorFlow 1.x based
- Reference for comparison

## Results

On NELL-995 dataset, DeepPath discovers interpretable reasoning paths:

**Example: athletePlaysForTeam**
- Path 1: athlete → playsPosition → position → positionPlayedByTeam → team
- Path 2: athlete → athleteHomeStadium → stadium → teamHomeStadium⁻¹ → team

These paths achieve competitive accuracy while providing explanations for predictions.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{xiong2017deeppath,
  title = {DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning},
  author = {Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2017},
  pages = {564--573}
}
```

## Acknowledgements

- [Original DeepPath implementation](https://github.com/xwhan/DeepPath)
- NELL-995 dataset from [KB-Reasoning-Data](https://github.com/wenhuchen/KB-Reasoning-Data)