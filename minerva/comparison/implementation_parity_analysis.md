# MINERVA Implementation Parity Analysis: TensorFlow vs PyTorch

This document provides a detailed comparison between the original TensorFlow implementation and our PyTorch implementation to ensure functional parity.

## 1. Agent Class Comparison

### Key Components

| Component | TensorFlow (original/code/model/agent.py) | PyTorch (src/model/agent.py) | Parity Status |
|-----------|-------------------------------------------|-------------------------------|---------------|
| **Embeddings** | | | |
| Relation Embeddings | `tf.get_variable` with shape `[action_vocab_size, 2*embedding_size]` | `nn.Embedding(action_vocab_size, 2*embedding_size)` | ✅ Equivalent |
| Entity Embeddings | `tf.get_variable` with shape `[entity_vocab_size, 2*embedding_size]` | `nn.Embedding(entity_vocab_size, 2*embedding_size)` | ✅ Equivalent |
| Initialization | `xavier_initializer()` for relations, `zeros_initializer()` for entities (when not using) | `xavier_uniform_` for relations, `zero_()` for entities (when not using) | ✅ Equivalent |
| **LSTM** | | | |
| Architecture | `MultiRNNCell` with `LSTMCell` layers | `nn.LSTM` with `num_layers` | ✅ Equivalent |
| Hidden Size | `m * hidden_size` where m=4 with entities, m=2 without | Same logic implemented | ✅ Equivalent |
| State Handling | `state_is_tuple=True` | Native PyTorch tuple state `(h, c)` | ✅ Equivalent |
| **Policy MLP** | | | |
| Layer 1 | `tf.layers.dense(state, 4*hidden_size, relu)` | `nn.Linear(input_size, 4*hidden_size)` + ReLU | ✅ Equivalent |
| Layer 2 | `tf.layers.dense(hidden, m*embedding_size, relu)` | `nn.Linear(4*hidden_size, m*embedding_size)` + ReLU | ✅ Equivalent |

### Action Selection Logic

| Feature | TensorFlow | PyTorch | Parity Status |
|---------|------------|---------|---------------|
| Score Computation | `tf.reduce_sum(tf.multiply(embeddings, output), axis=2)` | `torch.sum(embeddings * output, dim=2)` | ✅ Equivalent |
| PAD Masking | `tf.where(mask, dummy_scores, prelim_scores)` with -99999 | `masked_fill(mask, -99999.0)` | ✅ Equivalent |
| Action Sampling | `tf.multinomial(logits=scores, num_samples=1)` | `torch.multinomial(F.softmax(scores), 1)` | ✅ Equivalent |
| Log Probs | `tf.nn.log_softmax(scores)` | `F.log_softmax(scores, dim=1)` | ✅ Equivalent |

### Critical Differences Found

1. **Loss Computation Location**:
   - TF: Loss computed inside agent.step() using `sparse_softmax_cross_entropy_with_logits`
   - PyTorch: Loss computed in trainer using log_probs and actions
   - **Impact**: None - mathematically equivalent

2. **Forward Pass Structure**:
   - TF: Uses `__call__` with full trajectory
   - PyTorch: Uses `forward` with full trajectory + separate `step` function
   - **Impact**: None - same functionality

## 2. Environment Class Comparison

### Episode Management

| Component | TensorFlow (original/code/model/environment.py) | PyTorch (src/model/environment.py) | Parity Status |
|-----------|------------------------------------------------|-------------------------------------|---------------|
| Data Structure | NumPy arrays | NumPy arrays | ✅ Identical |
| Rollout Expansion | `np.repeat(entities, num_rollouts)` | `np.repeat(entities, num_rollouts)` | ✅ Identical |
| State Dictionary | `{'next_relations', 'next_entities', 'current_entities'}` | Same keys and structure | ✅ Identical |
| Reward Calculation | Boolean comparison then `np.select` | `np.where` with same logic | ✅ Equivalent |

### Environment Class

| Feature | TensorFlow | PyTorch | Parity Status |
|---------|------------|---------|---------------|
| Batcher Integration | Creates `RelationEntityBatcher` | Same initialization | ✅ Identical |
| Grapher Integration | Creates `RelationEntityGrapher` | Same initialization | ✅ Identical |
| Episode Generation | Generator pattern with `yield` | Same generator pattern | ✅ Identical |

## 3. Training Algorithm Comparison

### REINFORCE Implementation

| Component | TensorFlow (original/code/model/trainer.py) | PyTorch (src/model/trainer.py) | Parity Status |
|-----------|---------------------------------------------|--------------------------------|---------------|
| **Loss Calculation** | | | |
| Per-step Loss | Computed in agent, stacked in trainer | Computed in trainer from log_probs | ✅ Equivalent |
| Baseline | `ReactiveBaseline` with exponential average | Same implementation | ✅ Identical |
| Advantage | `(reward - baseline) / std` | Same normalization | ✅ Identical |
| Entropy Reg | `-β * mean(sum(exp(logits) * logits))` | Same formula | ✅ Identical |
| **Optimization** | | | |
| Optimizer | `AdamOptimizer` | `torch.optim.Adam` | ✅ Equivalent |
| Gradient Clipping | `clip_by_global_norm` | `clip_grad_norm_` | ✅ Equivalent |
| Learning Rate | Fixed | Fixed (same default) | ✅ Identical |

### Cumulative Discounted Reward

```python
# Both implementations use identical logic:
cum_disc_reward[:, T-1] = rewards
for t in reversed(range(T)):
    running_add = gamma * running_add + cum_disc_reward[:, t]
    cum_disc_reward[:, t] = running_add
```

✅ **Identical implementation**

## 4. Evaluation Metrics

### Beam Search

| Feature | TensorFlow | PyTorch | Parity Status |
|---------|------------|---------|---------------|
| Beam Size | `test_rollouts` (default 100) | Same parameter | ✅ Identical |
| Score Accumulation | Addition of log probs | Same method | ✅ Identical |
| Top-k Selection | Custom `top_k` function | PyTorch `topk` + reshaping | ✅ Equivalent |
| State Reordering | Manual indexing | Same logic with PyTorch tensors | ✅ Equivalent |

### Metrics Calculation

| Metric | TensorFlow | PyTorch | Parity Status |
|--------|------------|---------|---------------|
| Hits@k | Count positions < k | Same logic | ✅ Identical |
| MRR | `1.0 / (answer_pos + 1)` | Same formula | ✅ Identical |
| Answer Position | Deduplication + ranking | Same algorithm | ✅ Identical |

## 5. Data Loading Comparison

### Grapher

| Feature | TensorFlow | PyTorch | Parity Status |
|---------|------------|---------|---------------|
| Graph Loading | Dictionary of entity -> [(entity, relation)] | Same structure | ✅ Identical |
| Action Sampling | Random sampling when > max_actions | Same logic | ✅ Identical |
| Padding | PAD tokens for invalid actions | Same approach | ✅ Identical |

### Batcher

| Feature | TensorFlow | PyTorch | Parity Status |
|---------|------------|---------|---------------|
| Data Format | `[e1, r, e2]` triples | Same format | ✅ Identical |
| Answer Lookup | Dictionary of (e1, r) -> set of e2 | Same structure | ✅ Identical |
| Batch Generation | Shuffle + slice for train, sequential for test | Same logic | ✅ Identical |

## 6. Critical Implementation Details

### Verified Identical Behaviors

1. **Dummy Start Relation**: Both use `DUMMY_START_RELATION` token
2. **PAD Token Handling**: Same PAD tokens for entities and relations
3. **Reward Values**: Same positive (1.0) and negative (0) defaults
4. **Path Length**: Same default of 3 steps
5. **Batch Size Calculation**: `batch_size * num_rollouts`

### Minor Differences (No Impact on Results)

1. **Tensor Operations**: TF uses `tf.*` ops, PyTorch uses `torch.*` - mathematically equivalent
2. **State Management**: TF uses placeholders + session, PyTorch uses direct tensor ops
3. **Variable Scope**: TF uses `variable_scope`, PyTorch uses module hierarchy

## 7. Potential Issues to Address

### ⚠️ Areas Requiring Attention

1. **Random Seed Management**: 
   - TF version doesn't show explicit seed setting
   - PyTorch version should add seed management for reproducibility

2. **Pretrained Embeddings**:
   - TF loads from text files with `np.loadtxt`
   - PyTorch expects tensors - need to ensure same loading logic

3. **Device Placement**:
   - TF uses session config
   - PyTorch uses `.to(device)` - ensure consistent device usage

## Conclusion

The PyTorch implementation maintains functional parity with the TensorFlow version across all critical components:

- ✅ Agent architecture and policy network
- ✅ Environment and episode management  
- ✅ REINFORCE algorithm with baseline
- ✅ Evaluation metrics and beam search
- ✅ Data loading and preprocessing

The implementations are mathematically equivalent and should produce comparable results given the same hyperparameters and random seeds.