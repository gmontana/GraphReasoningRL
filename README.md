# Graph Reasoning with Reinforcement Learning

This repository contains implementations of reinforcement learning algorithms for reasoning over knowledge graphs.

## Algorithms Implemented

- **DeepPath**: Reasoning with Reinforcement Learning - [DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning](deeppath/) (EMNLP 2017)
  - PyTorch implementation
  - Path-finding approach using policy gradient

- **MINERVA**: Multi-Hop Reasoning - [Go for a Walk and Arrive at the Answer](minerva/) (ICLR 2018)
  - TensorFlow (original) and PyTorch implementations
  - Learns to navigate knowledge graphs by following relations

More algorithms may be added in the future.

## Repository Structure

```
GraphReasoningRL/
├── deeppath/          # DeepPath implementation
│   ├── src/           # PyTorch source code
│   ├── models/        # Trained models
│   └── demo/          # Demo and examples
├── minerva/           # MINERVA implementation
│   ├── original/      # Original TensorFlow code
│   ├── src/           # PyTorch implementation
│   └── comparison/    # Tools to compare implementations
└── README.md          # This file
```

## Getting Started

Each algorithm has its own directory with detailed documentation. Start with:
- [MINERVA Documentation](minerva/README.md)

## License

Please refer to individual algorithm directories for specific license information.