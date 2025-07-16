# DeepPath Implementation Comparison

This directory contains tools for comparing the PyTorch and TensorFlow implementations of DeepPath.

## Overview

DeepPath is implemented in two versions:
- PyTorch implementation (in `/deeppath` directory)
- Original TensorFlow implementation (in `/comparison/tensorflow` directory)

The full implementation includes:
1. Path-finding using reinforcement learning with the REINFORCE algorithm
2. Path evaluation using Mean Average Precision (MAP)
3. Training a prediction model on discovered paths

## Benchmark Metrics

When comparing the PyTorch and TensorFlow implementations, we measure:

1. **Runtime Performance**
   - Total runtime for path finding
   - Hardware acceleration effectiveness (MPS/CUDA)

2. **Path Discovery Quality**
   - Number of paths discovered
   - Success rate in finding paths
   - Average path length
   - Path diversity and relevance

3. **Evaluation Metrics**
   - Mean Average Precision (MAP)
   - Path statistics distribution

## Running the Complete Benchmark

To run the complete benchmark that executes both implementations:

```bash
# From the comparison directory
./run_complete_benchmark.sh athletePlaysForTeam
```

This script will:
1. Create and use dedicated conda environments for both implementations
2. Run the PyTorch implementation (in `deeppath_torch` environment)
3. Run the TensorFlow implementation (in `deeppath_tf2` environment)
4. Run evaluation on both results
5. Compare the implementations
6. Generate a comprehensive benchmark report

### Benchmark Dataset Options

We provide two different benchmark datasets:

1. **Simple Benchmark Dataset**:
   ```bash
   ./create_benchmark_dataset.sh
   ```
   Creates a small, controlled dataset with direct paths between entities.

2. **Complex Benchmark Dataset**:
   ```bash
   ./create_complex_benchmark.sh
   ```
   Creates a rich knowledge graph with many entity types and relationships, forcing the implementations to find longer multi-hop paths between entities by removing direct connections.

3. **Full NELL-995 Dataset**:
   ```bash
   ./setup_full_dataset.sh
   ```
   Downloads the complete NELL-995 dataset from [wenhuchen/KB-Reasoning-Data](https://github.com/wenhuchen/KB-Reasoning-Data) and sets it up for comprehensive benchmarking with real-world data.

## Conda Environments

The benchmark scripts create and use dedicated conda environments:

1. **deeppath_torch**: For the PyTorch implementation
   - Created by `setup_conda_env.sh` in the repository root
   - Uses PyTorch with appropriate hardware acceleration
   - Python 3.10 with PyTorch, NumPy, SciPy, and scikit-learn
   - Apple Silicon support via MPS backend

2. **deeppath_tf2**: For the TensorFlow implementation
   - Created by `run_tf_modern.sh`
   - Uses TensorFlow 2.x with compatibility layer for TF1.x code
   - Python 3.8 with TensorFlow 2.x, NumPy, scikit-learn, and Keras

These environments ensure proper isolation and dependency management.

### Maintaining Conda Environments

If you need to update or recreate these environments:

```bash
# Remove and recreate the PyTorch environment
conda remove -n deeppath_torch --all -y
./setup_conda_env.sh

# Remove and recreate the TensorFlow environment
conda remove -n deeppath_tf2 --all -y
# The environment will be recreated when running run_tf_modern.sh
```

The benchmark scripts will automatically check for the existence of these environments and create them if needed.

## TensorFlow 2.x Compatibility

We've added TensorFlow 2.x compatibility to run the original TensorFlow 1.x code on modern systems:

1. `tensorflow/tf2_compatibility.py` - Creates a compatibility layer that allows TensorFlow 1.x code to run in TensorFlow 2.x
2. `run_tf_modern.sh` - Script to run the TensorFlow implementation with the compatibility layer
3. `tensorflow/evaluate_simple.py` - Simplified evaluation script that doesn't rely on keras for better compatibility

Key changes for TensorFlow 2.x compatibility:
- Enabling eager execution
- Using TensorFlow's compatibility API (tf.compat.v1)
- Mocking removed modules like `tf.contrib` to avoid dependency on keras
- Adding safety checks for out-of-range action indices
- Fixing action space size to match the actual number of relations
- Python 3 syntax updates for the original code
- Using absolute paths for better file handling reliability

## Running Individual Components

### Path Finding

To run just the path-finding component:

```bash
# PyTorch Implementation (from repository root)
python main.py athletePlaysForTeam

# TensorFlow Implementation (from benchmarks directory)
./run_tf_modern.sh athletePlaysForTeam
```

### Evaluation

After running path-finding, evaluate the quality of discovered paths:

```bash
# From the repository root
python run_evaluation.py athletePlaysForTeam
```

### Implementation Comparison

To compare the PyTorch and TensorFlow implementations:

```bash
# From the comparison directory
python compare_implementations.py athletePlaysForTeam
```

## Available Relations

Benchmark any of these relations:
- `athletePlaysForTeam` (default test case)
- `athletePlaysInLeague` 
- `athleteHomeStadium`
- `teamPlaySports`
- `organizationHeadquarteredInCity`

## Benchmark Results

After running benchmarks, results are available in:
- `logs/pytorch_*.log` - PyTorch implementation logs
- `logs/tensorflow_*.log` - TensorFlow implementation logs
- `logs/comparison_*.log` - Implementation comparison results

### Understanding Path Files

The benchmark produces several path-related files:

1. **path_to_use.txt**: Contains the discovered paths
   ```
   teamPlaySport -> athletePlaySport -> ~athletePlaysForTeam
   athleteInLeague -> teamPlaysinLeague -> ~athletePlaysForTeam
   ```

2. **path_stats.txt**: Contains path frequency statistics
   ```
   teamPlaySport -> athletePlaySport -> ~athletePlaysForTeam    5
   athleteInLeague -> teamPlaysinLeague -> ~athletePlaysForTeam    3
   ```

3. **path_evaluation.txt**: Contains evaluation metrics
   ```
   Paths found: 10
   Average path length: 2.80
   Success rate: 0.65
   ```

## Files in this Directory

- `README.md` - This documentation file
- `run_complete_benchmark.sh` - Comprehensive benchmark script for both implementations
- `run_tf_modern.sh` - Script to run TensorFlow with compatibility layer
- `create_benchmark_dataset.sh` - Script to create a simple benchmark dataset
- `create_complex_benchmark.sh` - Script to create a complex benchmark dataset with longer paths
- `setup_full_dataset.sh` - Script to download and set up the full NELL-995 dataset
- `analyze_benchmark_results.py` - Script to analyze benchmark results
- `compare_implementations.py` - Script to compare implementation outputs
- `run_evaluation.py` - Script to evaluate path quality using MAP
- `tensorflow/` - Original TensorFlow implementation with compatibility fixes
- `logs/` - Directory for benchmark logs and results

## Expected Performance

When comparing the implementations:

1. **Path Finding**:
   - Both implementations should find valid paths
   - Runtime performance varies based on hardware and configuration
   - Path diversity may differ between implementations

2. **Complex Path Discovery**:
   - Different implementations may discover different path patterns
   - Path variety depends on random initialization and algorithmic details

3. **Evaluation**:
   - Both implementations should achieve comparable Mean Average Precision (MAP)
   - Success rates may vary depending on the knowledge graph complexity

## Full Evaluation Pipeline

The complete DeepPath pipeline consists of:

1. **Path Finding**: Finding paths between entity pairs using reinforcement learning
2. **Path Analysis**: Analyzing discovered paths and saving path statistics
3. **Evaluation**: Training an evaluation model and calculating MAP
4. **Implementation Comparison**: Comparing results between PyTorch and TensorFlow

## Hardware Requirements

- Conda (tested with 25.1.1+)
- NELL-995 dataset (included in repository or downloadable with `setup_full_dataset.sh`)
- At least 4GB RAM
- For PyTorch:
  - Apple Silicon support via MPS backend
  - CUDA support for NVIDIA GPUs
- For TensorFlow:
  - TensorFlow 2.x with compatibility mode

## Analyzing Differences

When implementations produce different results, analyze:

1. **Path Discovery Differences**:
   - Check common paths vs. unique paths
   - Compare success rates between implementations
   - Analyze path statistics and distributions

2. **Performance Differences**:
   - Runtime comparison
   - Hardware acceleration effectiveness

3. **Algorithm Implementation**:
   - REINFORCE algorithm implementation differences
   - Reward calculation variations
   - Action space handling

## Troubleshooting

- **Conda Environment Issues**: If conda environments fail to create or activate, try running `conda init` and restarting your shell
- **TensorFlow Compatibility**: If TensorFlow compatibility issues occur, check TensorFlow version and compatibility layer settings
- **Dataset Issues**: If the dataset isn't loading, verify the NELL-995 directory structure or run `./setup_full_dataset.sh` to set up the complete dataset
- **Missing Output Files**: Ensure that the path_to_use.txt and path_stats.txt files are generated in the appropriate relation directory
- **Keras-Related Errors**: If the original evaluation fails with keras errors, the simplified evaluation script (`evaluate_simple.py`) will still run successfully
- **Index Out of Range Errors**: Fixed with safety checks in the TensorFlow environment to prevent crashes
- **Path Issues**: If you see path-related errors, check that absolute paths are being used correctly for your system
- **Memory Errors**: If you encounter memory issues, try reducing the action space size in `tensorflow/utils.py`