# DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning

DeepPath is a reinforcement learning method for interpretable reasoning over knowledge graphs. It finds multi-hop paths between entities by training an agent to navigate from source to target entities, providing explanations for knowledge graph relations.

## Overview

- **Algorithm**: Combines supervised learning (imitation from a BFS teacher) and reinforcement learning (REINFORCE) to discover diverse reasoning paths.
- **Input**: (source_entity, target_entity, relation_type)
- **Output**: Ranked paths connecting source to target, explaining the relation.
- **Key Features**: Path diversity, interpretable reasoning, two-phase training, evaluation via Mean Average Precision (MAP).

For a detailed algorithm description, see the original paper and the [src/README.md](src/README.md) for PyTorch-specific details and configuration.

## Reinforcement Learning Setup

### Task
The agent is tasked with finding multi-hop reasoning paths in a knowledge graph that connect a source entity to a target entity via a specified relation. The goal is to discover interpretable paths that explain why a relation holds between two entities.

### States
Each state consists of the current entity’s embedding and the target entity’s embedding, concatenated into a fixed-size vector. This provides the agent with information about its current position and the goal.

### Actions
At each step, the agent selects a relation to follow from the current entity. The next entity is determined by the knowledge graph structure. Invalid actions (relations not present from the current entity) are masked out.

### Rewards
- **Terminal reward**: +1 if the agent reaches the target entity at the end of the path, 0 otherwise.
- **Efficiency bonus**: Shorter paths are rewarded (e.g., 1/path_length).
- **Diversity bonus**: Additional reward for discovering novel paths.
- **Global reward**: Based on accuracy@1 on the validation set.

### Learning Algorithm
- **Phase 1**: Supervised learning from a BFS teacher, using cross-entropy loss to imitate correct paths.
- **Phase 2**: Policy gradient reinforcement learning (REINFORCE with baseline), encouraging exploration and path diversity.

### Embeddings
- Entity and relation embeddings are loaded from local files (`entity2vec.bern`, `relation2vec.bern`) if available, or randomly initialized otherwise.
- These embeddings are fixed (not updated during training) and are used to construct the state vector (current entity embedding + target entity embedding) for the policy network.
- Embedding dimension is configurable (default: 100).

## Implementation Notes

- **Action masking**: Invalid actions (relations not present from the current entity) are masked out during action selection, ensuring the agent only chooses valid transitions.
- **Model architecture**: The policy, value, and Q-networks are multi-layer perceptrons (MLPs), not LSTMs.
- **Direct link removal**: During supervised training, the BFS teacher removes direct links between source and target entities to encourage the agent to find indirect reasoning paths.
- **File/data requirements**: The code expects certain files in the dataset directory, including `entity2vec.bern`, `relation2vec.bern`, `entity2id.txt`, `relation2id.txt`, and `kb_env_rl.txt`. Ensure these are present for correct operation.
- **Handling minimal/test datasets**: If embedding files are too small or malformed, the code pads or truncates embeddings as needed to match the expected dimensions.

## Installation

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Dataset

DeepPath uses the NELL-995 dataset. Download it from:
- [KB-Reasoning-Data](https://github.com/wenhuchen/KB-Reasoning-Data)

```bash
git clone https://github.com/wenhuchen/KB-Reasoning-Data.git
cp -r KB-Reasoning-Data/NELL-995 ./
```

## Usage (PyTorch)

Run the full pipeline (supervised + RL + evaluation):

```bash
python main.py athletePlaysForTeam
```

For individual phases (see `src/README.md` for all options):

```bash
python main.py athletePlaysForTeam --mode train      # Train (supervised + RL)
python main.py athletePlaysForTeam --mode test       # Evaluation only
```

## Comparing Implementations

To compare PyTorch and TensorFlow results:

```bash
cd comparison
python compare_implementations.py athletePlaysForTeam
./run_complete_benchmark.sh athletePlaysForTeam
```

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
