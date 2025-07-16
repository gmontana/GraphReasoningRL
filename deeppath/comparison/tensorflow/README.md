# DeepPath TensorFlow Version

This directory contains the original TensorFlow implementation of DeepPath.

## Contents

- `networks.py` - Neural network implementations in TensorFlow
- `policy_agent.py` - Policy agent for reinforcement learning
- `sl_policy.py` - Supervised learning policy
- `env.py` - Environment for interacting with knowledge graphs
- `utils.py` - Utility functions
- `evaluate.py` - Evaluation script
- `transE_eval.py`, `transR_eval.py` - TransE and TransR evaluations
- `fact_prediction_eval.py` - Fact prediction evaluation
- `BFS/` - Breadth-first search implementation
- `pathfinder.sh` - Script to run training and evaluation
- `link_prediction_eval.sh` - Script for link prediction evaluation

## Requirements

- TensorFlow 1.x
- NumPy
- scikit-learn
- Keras

## Usage

Run the pathfinder script with a relation:

```
./pathfinder.sh athletePlaysForTeam
```

This will:
1. Train a supervised policy (SL)
2. Retrain with reinforcement learning (RL)
3. Test the trained policy