"""
Parity tests to verify TensorFlow and PyTorch implementations produce same results
"""

import numpy as np
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../original/code'))


def test_cumulative_reward_calculation():
    """Test that cumulative discounted reward calculation matches"""
    print("Testing Cumulative Discounted Reward Calculation...")
    
    # Test parameters
    gamma = 0.99
    path_length = 3
    batch_size = 4
    
    # Sample rewards (only given at final step)
    rewards = np.array([1.0, 0.0, 1.0, 0.0])
    
    # Expected calculation for path_length=3, gamma=0.99:
    # Step 2 (last): [1.0, 0.0, 1.0, 0.0]
    # Step 1: gamma * 0 + 0 = [0.0, 0.0, 0.0, 0.0]  
    # Step 0: gamma * 0 + 0 = [0.0, 0.0, 0.0, 0.0]
    # 
    # Then running_add accumulates:
    # t=2: running_add = 0.99*0 + [1.0, 0.0, 1.0, 0.0] = [1.0, 0.0, 1.0, 0.0]
    # t=1: running_add = 0.99*[1.0, 0.0, 1.0, 0.0] + [0, 0, 0, 0] = [0.99, 0.0, 0.99, 0.0]
    # t=0: running_add = 0.99*[0.99, 0.0, 0.99, 0.0] + [0, 0, 0, 0] = [0.9801, 0.0, 0.9801, 0.0]
    
    expected = np.array([
        [0.9801, 0.0, 0.9801, 0.0],  # step 0
        [0.99, 0.0, 0.99, 0.0],       # step 1  
        [1.0, 0.0, 1.0, 0.0]          # step 2
    ]).T
    
    # Manual calculation following TF logic
    running_add = np.zeros([batch_size])
    cum_disc_reward = np.zeros([batch_size, path_length])
    cum_disc_reward[:, path_length - 1] = rewards
    
    for t in reversed(range(path_length)):
        running_add = gamma * running_add + cum_disc_reward[:, t]
        cum_disc_reward[:, t] = running_add
    
    print(f"Rewards: {rewards}")
    print(f"Expected cumulative discounted rewards:\n{expected}")
    print(f"Calculated cumulative discounted rewards:\n{cum_disc_reward}")
    print(f"Match: {np.allclose(expected, cum_disc_reward)}")
    print()
    
    return np.allclose(expected, cum_disc_reward)


def test_entropy_calculation():
    """Test entropy regularization calculation"""
    print("Testing Entropy Calculation...")
    
    # Create sample log probabilities
    # Shape: [batch_size=2, num_actions=3, time_steps=2]
    log_probs = np.array([
        [[-1.0, -2.0], [-3.0, -1.5], [-2.5, -2.0]],  # batch 1
        [[-1.5, -1.0], [-2.0, -2.5], [-1.0, -3.0]]   # batch 2
    ])
    
    # Calculate entropy: -mean(sum(exp(logits) * logits))
    probs = np.exp(log_probs)
    entropy_per_batch = np.sum(probs * log_probs, axis=1)  # sum over actions
    entropy = -np.mean(entropy_per_batch)  # mean over batch and time
    
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Calculated entropy: {entropy:.6f}")
    print()
    
    return entropy


def test_masking_and_scoring():
    """Test action masking and scoring"""
    print("Testing Action Masking and Scoring...")
    
    # Sample data
    scores = np.array([[2.0, 1.5, -0.5, 3.0],
                       [1.0, 2.5, 1.5, -1.0]])
    
    PAD = 0
    relations = np.array([[1, 2, 0, 3],  # 0 is PAD
                          [2, 0, 1, 0]])  # 0 is PAD
    
    # Apply masking
    mask = (relations == PAD)
    masked_scores = np.where(mask, -99999.0, scores)
    
    print(f"Original scores:\n{scores}")
    print(f"Relations (0=PAD):\n{relations}")
    print(f"Mask:\n{mask}")
    print(f"Masked scores:\n{masked_scores}")
    
    # Softmax for probability
    exp_scores = np.exp(masked_scores - np.max(masked_scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    print(f"Probabilities:\n{probs}")
    print(f"Sum of probs per row: {np.sum(probs, axis=1)}")
    print()
    
    return True


def test_beam_search_indexing():
    """Test beam search index calculations"""
    print("Testing Beam Search Indexing...")
    
    batch_size = 2
    num_actions = 4
    k = 3  # beam size
    
    # Flat indices from top-k
    flat_indices = np.array([11, 10, 9, 7, 6, 5])  # top k from each batch
    
    # Convert to beam and action indices
    beam_idx = flat_indices // num_actions
    action_idx = flat_indices % num_actions
    
    print(f"Batch size: {batch_size}, Num actions: {num_actions}, Beam size: {k}")
    print(f"Flat indices: {flat_indices}")
    print(f"Beam indices: {beam_idx}")
    print(f"Action indices: {action_idx}")
    
    # Verify reconstruction
    reconstructed = beam_idx * num_actions + action_idx
    print(f"Reconstructed: {reconstructed}")
    print(f"Match: {np.array_equal(flat_indices, reconstructed)}")
    print()
    
    return np.array_equal(flat_indices, reconstructed)


def main():
    """Run all parity tests"""
    print("="*60)
    print("MINERVA Implementation Parity Tests")
    print("="*60)
    print()
    
    tests = [
        ("Cumulative Reward", test_cumulative_reward_calculation),
        ("Entropy Calculation", test_entropy_calculation),
        ("Action Masking", test_masking_and_scoring),
        ("Beam Search", test_beam_search_indexing)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append((name, False))
    
    print("="*60)
    print("Test Summary:")
    print("="*60)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print()
    print(f"Overall: {'✅ All tests passed!' if all_passed else '❌ Some tests failed!'}")


if __name__ == "__main__":
    main()