# DeepPath: Reinforcement Learning for Knowledge Graph Reasoning

This repository contains a PyTorch implementation of DeepPath, a reinforcement learning approach for knowledge graph reasoning. This implementation introduces several modifications to the original TensorFlow version, including native hardware acceleration support, filtered knowledge graph training, improved path diversity calculation, and robust error handling while maintaining functional equivalence.

## How DeepPath Works

### 1. Knowledge Graph and Problem Formulation

A knowledge graph (KG) is a structured representation of facts in the form of entities and relations between them. Each fact is a triple (head entity, relation, tail entity), such as (Michael Jordan, playsFor, Chicago Bulls). 

The task of knowledge graph reasoning is to infer missing relations between entities. Instead of simply memorizing patterns from data, we want to understand the reasoning paths that explain why certain relations exist. This enables:
- Better explainability of predictions
- Higher accuracy by using multi-hop reasoning
- Generalization to unseen entity pairs

DeepPath approaches this task by finding meaningful reasoning paths between entities. For example, given a relation like "athletePlaysForTeam", the algorithm might discover paths such as:
- athletePlaysSport → sportPlayedByTeam (An athlete plays a sport that is played by a team)
- hasCoach → coachesTeam (An athlete has a coach who coaches a team)

### 2. Agent-Environment System

- **State**: Entity pair embeddings (current entity + target entity)
- **Actions**: Selecting which relation to follow from the current entity
- **Environment**: The knowledge graph (excluding direct links between query entities)
- **Reward**: Combination of:
  - Success reward (when target is reached)
  - Efficiency reward (inversely proportional to path length)
  - Diversity reward (finding diverse reasoning patterns)

### 3. Learning Process

1. **Supervised Phase**: The agent first learns from a "teacher" that uses BFS to find valid paths
2. **Reinforcement Phase**: The agent then explores on its own, refining its policy through REINFORCE algorithm
3. **Path Ranking**: Finally, the discovered paths are ranked by their efficiency and predictive power

### 4. Path Evaluation

The discovered paths are evaluated based on their ability to answer new entity pair queries, where no direct relation edge exists. The algorithm uses the Mean Average Precision (MAP) metric to evaluate path quality.

## Installation

### Option 1: Install as a Python package (Development Mode)

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepPath.git
cd DeepPath

# Install in development mode
pip install -e .
```

This installs the package in development mode, allowing you to modify the code and have changes immediately available.

### Option 2: Using Conda environment

```bash
# Run the setup script
./setup_conda_env.sh

# Activate the environment
conda activate deeppath_torch

# Install the package in development mode
pip install -e .
```

### Option 3: Using pip requirements only

```bash
pip install -r requirements.txt
```

Note: This only installs dependencies and doesn't install the DeepPath package itself.

## Download Dataset

Download the knowledge graph dataset from the official repository:
- [NELL-995](https://github.com/wenhuchen/KB-Reasoning-Data/tree/master/NELL-995)

You can clone the dataset repository directly:

```bash
git clone https://github.com/wenhuchen/KB-Reasoning-Data.git
cp -r KB-Reasoning-Data/NELL-995 ./
```

Extract the dataset into the repository directory.

## Usage

### Training and Testing

You can run the full pipeline using the main script:

```bash
# Run full pipeline (train + test)
python main.py athletePlaysForTeam

# Training only
python main.py athletePlaysForTeam --mode train

# Testing only
python main.py athletePlaysForTeam --mode test
```

Or use the convenience script:

```bash
./pathfinder.sh athletePlaysForTeam
```

### Core Components

The DeepPath implementation includes these core components:

- **Environment**: Manages the knowledge graph and handles state transitions
- **Agent**: Implements the policy network for path finding
- **Search**: Provides the teacher algorithm for supervised learning 
- **Utils**: Contains utility functions and constants

### Code Structure

```
DeepPath/
├── deeppath/               # Main Python package
│   ├── __init__.py         # Package initialization
│   ├── agents.py           # Agent implementations
│   ├── environment.py      # Knowledge graph environment
│   ├── evaluate.py         # Evaluation utilities
│   ├── models.py           # Neural network models
│   ├── search.py           # Path finding algorithms
│   └── utils.py            # Utility functions
├── main.py                 # CLI entry point
├── pathfinder.sh           # Convenience script
├── setup.py                # Package setup script
├── pyproject.toml          # Project metadata
└── requirements.txt        # Package dependencies
```

### Implementation Notes

- **Path Finding**: The teacher algorithm removes direct links between entities to encourage finding meaningful indirect paths
- **Reinforcement Learning**: Uses the REINFORCE algorithm to train the policy network
- **Evaluation**: Paths are evaluated based on efficiency (path length) and diversity

## Implementation Details

Our PyTorch implementation maintains functional equivalence with the original algorithm:
- Same REINFORCE algorithm with teacher guidance
- Identical state representation and network architecture
- Comparable success rates and Mean Average Precision (MAP) in evaluation

## Citation

If you use this code, please cite the original paper:

```
@InProceedings{wenhan_emnlp2017,
  author    = {Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
  title     = {DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017)},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {ACL}
}
```

## Acknowledgements

- [Original DeepPath implementation (TensorFlow)](https://github.com/xwhan/DeepPath)