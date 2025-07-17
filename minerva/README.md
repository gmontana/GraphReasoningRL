# MINERVA: Meandering In Networks of Entities to Reach Verisimilar Answers

MINERVA is a reinforcement learning agent for knowledge graph reasoning, implemented in both TensorFlow (original, modernized) and PyTorch (feature-complete, multi-device).

## Overview

- **Algorithm**: Sequential decision process for multi-hop reasoning in knowledge graphs.
- **Input**: (source_entity, query_relation, ?)
- **Output**: target_entity via path reasoning.
- **Features**: LSTM policy, beam search, variance reduction, multi-device support.

For detailed algorithm and PyTorch implementation, see [src/README.md](src/README.md).

## Reinforcement Learning Setup

### Task
The agent answers queries of the form (source entity, query relation, ?) by navigating the knowledge graph to reach the correct target entity, using a sequence of relation-following steps.

### States
The state at each step is a tuple (current entity, path history), where the path history is encoded by an LSTM. This allows the agent to condition its decisions on the sequence of relations and entities traversed so far.

### Actions
The agent chooses a (relation, next entity) pair from the set of valid outgoing edges of the current entity. Self-loops (NO_OP actions) are included to allow the agent to remain at the current node if needed.

### Rewards
- **Sparse terminal reward**: +1 if the agent ends at the correct target entity, 0 otherwise.
- No intermediate rewards are given.

### Learning Algorithm
- **Policy gradient (REINFORCE)**: The agent samples action sequences, receives the terminal reward, and updates its policy using the REINFORCE algorithm with a moving average baseline for variance reduction.
- **Entropy regularization**: Encourages exploration by penalizing low-entropy (overconfident) policies.

### Embeddings
- Relation and entity embeddings are implemented as `nn.Embedding` layers in PyTorch.
- You can choose to use entity embeddings, and to train or freeze entity/relation embeddings via configuration options.
- Pretrained embeddings can be loaded for both entities and relations; otherwise, embeddings are initialized with Xavier uniform.
- Embedding size is configurable (default: 50).

## Implementation Notes

- **Action masking**: Invalid actions (PAD) are masked with large negative values during action selection, ensuring the agent only chooses valid transitions.
- **Model architecture**: The policy network is an LSTM (not an MLP), allowing the agent to encode path history.
- **NO_OP/self-loop actions**: Self-loop (NO_OP) actions are included for all entities, allowing the agent to remain at the current node if needed.
- **Batching and rollouts**: Multiple rollouts are performed per query during training and evaluation. Incomplete batches are skipped during training.
- **Special tokens in vocabularies**: The vocabularies include special tokens such as PAD, UNK, DUMMY_START_RELATION, and NO_OP, which are used throughout the code.
- **File/data requirements**: The code expects vocabulary files (`entity_vocab.json`, `relation_vocab.json`), `graph.txt`, and preprocessed data files in the dataset directory.
- **Reward normalization/advantage calculation**: A moving average baseline is used for variance reduction in policy gradient updates.
- **Gradient clipping and regularization**: Both are configurable via command-line or config options.

## Installation

### PyTorch Implementation

Install dependencies with pip:

```bash
pip install -r requirements_pytorch.txt
```

Or manually:

```bash
pip install torch numpy tqdm
```

### TensorFlow Implementation (Modernized)

```bash
pip install tensorflow>=2.10.0 numpy tqdm
```

For the legacy code, see `original/requirements.txt` (TensorFlow 1.3, Python 2.7).

## Datasets

Datasets are in `original/datasets/data_preprocessed/`. See [src/README.md](src/README.md) for preprocessing and vocabulary creation.

## Usage

### PyTorch

```bash
cd src/
python train.py ../configs/countries_s1.json
```

See [src/README.md](src/README.md) for advanced options and manual configuration.

### TensorFlow

```bash
cd comparison/
python run_tensorflow.py --dataset countries_S1 --iterations 100
```

## Comparing Implementations

```bash
cd comparison/
python compare.py --dataset countries_S1 --iterations 100
```

See `comparison/COMPARISON_GUIDE.md` and `comparison/PARITY_REPORT.md` for detailed analysis.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{minerva,
  title = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and Vilnis, Luke and Durugkar, Ishan and Krishnamurthy, Akshay and Smola, Alex and McCallum, Andrew},
  booktitle = {ICLR},
  year = {2018}
}
```

## Acknowledgements

- [Original MINERVA implementation (TensorFlow)](https://github.com/shehzaadzd/MINERVA)
- Datasets from the original repository