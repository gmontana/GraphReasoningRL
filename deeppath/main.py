#!/usr/bin/env python
"""
DeepPath - Main entry point for training and evaluating the PyTorch implementation.

This script provides a command-line interface for:
1. Training the DeepPath agent with supervised learning
2. Training the agent with reinforcement learning
3. Testing the agent's performance on knowledge graph reasoning tasks

Example usage:
    # Run full pipeline (train + test)
    python main.py athletePlaysForTeam
    
    # Training only
    python main.py athletePlaysForTeam --mode train
    
    # Testing only
    python main.py athletePlaysForTeam --mode test
    
    # Specify a custom dataset path
    python main.py athletePlaysForTeam --data_path ./custom/path/NELL-995/
"""

import os
import sys
import argparse
import time

from src.agents import PolicyAgent, SupervisedPolicyAgent, train_reinforce
from src.environment import Env
from src.utils import select_device

# Default data path
DEFAULT_DATA_PATH = './NELL-995/'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeepPath PyTorch Implementation')
    parser.add_argument('relation', type=str, help='The relation to train on')
    parser.add_argument('--mode', type=str, default='train_test', 
                      choices=['train', 'test', 'train_test'], 
                      help='Mode: train, test, or both')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                      help='Path to the NELL-995 dataset')
    parser.add_argument('--max_episodes', type=int, default=300,
                      help='Maximum number of episodes to train')
    return parser.parse_args()

def train_supervised(relation, data_path):
    """Train the agent with supervised learning."""
    print(f"\n===== Training Supervised Agent for Relation: {relation} =====")
    agent = SupervisedPolicyAgent()
    
    # Set up paths
    graph_path = os.path.join(data_path, 'tasks', relation, 'graph.txt')
    relation_path = os.path.join(data_path, 'tasks', relation, 'train_pos')
    
    # Create a training graph without direct links to encourage finding indirect paths
    train_graph_dir = os.path.join(data_path, 'tasks', relation, 'train_graph')
    train_graph_path = os.path.join(train_graph_dir, 'train_graph.txt')
    
    # Create the directory if it doesn't exist
    os.makedirs(train_graph_dir, exist_ok=True)
    
    # Create a filtered graph without direct links for the target relation
    if not os.path.exists(train_graph_path):
        print(f"Creating training graph without direct {relation} links...")
        relation_pattern = f"concept:{relation.lower()}"
        with open(graph_path, 'r') as f_in, open(train_graph_path, 'w') as f_out:
            for line in f_in:
                if relation_pattern not in line.lower():
                    f_out.write(line)
        print(f"Training graph created at {train_graph_path}")
    
    # Use the training graph for finding paths
    search_graph_path = train_graph_path
    
    # Read training data
    with open(relation_path) as f:
        train_data = f.readlines()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Limit the number of training samples if too many
    num_samples = len(train_data)
    if num_samples > 500:
        num_samples = 500
    
    # Training loop
    for episode in range(num_samples):
        print(f"Episode {episode}/{num_samples}")
        print(f'Training Sample: {train_data[episode % num_samples][:-1]}')
        
        env = Env(data_path, train_data[episode % num_samples])
        sample = train_data[episode % num_samples].split()
        
        # Get teaching examples
        try:
            from deeppath.search import teacher
            # Try with the filtered graph first
            good_episodes = teacher(sample[0], sample[1], 5, env, search_graph_path)
            
            # Fall back to the full graph if no paths found
            if not good_episodes:
                print("No paths found in filtered graph, trying full graph...")
                good_episodes = teacher(sample[0], sample[1], 5, env, graph_path)
                
            if not good_episodes:
                print("No paths found in either graph")
                continue
                
            print(f"Found {len(good_episodes)} teaching paths")
        except Exception as e:
            print('Cannot find a path:', e)
            continue
        
        # Update the agent with teaching examples
        for item in good_episodes:
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)
                action_batch.append(transition.action)
            import numpy as np
            state_batch = np.squeeze(state_batch)
            state_batch = np.reshape(state_batch, [-1, 200])  # state_dim = 200
            agent.update(state_batch, action_batch)
    
    # Save the model
    agent.save(f'models/policy_supervised_{relation}.pt')
    print('Supervised model saved')
    return agent

def train_rl(relation, data_path, max_episodes=300):
    """Train the agent with reinforcement learning."""
    print(f"\n===== Training RL Agent for Relation: {relation} =====")
    agent = PolicyAgent()
    
    # Check if supervised model exists and load it
    model_path = f'models/policy_supervised_{relation}.pt'
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Supervised policy loaded")
    else:
        print("No pre-trained model found. Please run in 'train' mode first")
        return None
    
    graph_path = os.path.join(data_path, 'tasks', relation, 'graph.txt')
    relation_path = os.path.join(data_path, 'tasks', relation, 'train_pos')
    
    # Read training data
    with open(relation_path) as f:
        train_data = f.readlines()
    
    # Create environment factory
    env_factory = lambda sample: Env(data_path, sample)
    
    # Limit the number of episodes if too many
    episodes = len(train_data)
    if episodes > max_episodes:
        episodes = max_episodes
    
    # Train with REINFORCE
    success, path_stats = train_reinforce(agent, train_data, env_factory, graph_path, episodes)
    
    # Save the retrained model
    agent.save(f'models/policy_retrained_{relation}.pt')
    print('RL model saved')
    
    # Save path statistics
    f = open(os.path.join(data_path, 'tasks', relation, 'path_stats.txt'), 'w')
    for item in path_stats:
        f.write(item[0]+'\t'+str(item[1])+'\n')
    f.close()
    print('Path stats saved')
    
    return agent

def test(relation, data_path):
    """Test the trained agent."""
    print(f"\n===== Testing Agent for Relation: {relation} =====")
    agent = PolicyAgent()
    
    # Check if RL model exists and load it
    model_path = f'models/policy_retrained_{relation}.pt'
    if os.path.exists(model_path):
        agent.load(model_path)
        print("RL policy loaded")
    else:
        print("No retrained model found. Please run in 'train' mode first")
        return
    
    relation_path = os.path.join(data_path, 'tasks', relation, 'train_pos')
    
    # Read test data
    with open(relation_path) as f:
        test_data = f.readlines()
    
    # Limit the number of test samples
    test_num = len(test_data)
    if test_num > 500:
        test_num = 500
    
    success = 0
    path_found = []
    
    for episode in range(test_num):
        print(f'Test sample {episode}/{test_num}: {test_data[episode][:-1]}')
        env = Env(data_path, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
        
        # Run the episode
        from itertools import count
        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = agent.predict(state_vec)
            
            action_probs = action_probs.squeeze()
            
            import numpy as np
            from deeppath.utils import ACTION_SPACE
            action_chosen = np.random.choice(np.arange(ACTION_SPACE), p=action_probs)
            reward, new_state, done = env.interact(state_idx, action_chosen)
            
            if done or t == 50:  # max_steps_test = 50
                if done:
                    success += 1
                    print("Success")
                    from deeppath.utils import path_clean
                    path = path_clean(' -> '.join(env.path))
                    path_found.append(path)
                else:
                    print('Episode ends due to step limit')
                break
            state_idx = new_state
    
    print(f'Success rate: {success}/{test_num} = {success/test_num:.2%}')
    
    # Analyze and save paths
    from collections import Counter
    import numpy as np
    
    path_relation_found = []
    for path in path_found:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))
    
    relation_path_stats = Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x:x[1], reverse=True)
    
    ranking_path = []
    for item in relation_path_stats:
        path = item[0]
        length = len(path.split(' -> '))
        ranking_path.append((path, length))
    
    ranking_path = sorted(ranking_path, key=lambda x:x[1])
    
    # Save paths to use
    f = open(os.path.join(data_path, 'tasks', relation, 'path_to_use.txt'), 'w')
    for item in ranking_path:
        f.write(item[0] + '\n')
    f.close()
    print('Path to use saved')

def main():
    """Main entry point."""
    args = parse_args()
    relation = args.relation
    mode = args.mode
    data_path = args.data_path
    max_episodes = args.max_episodes
    
    # Print hardware info
    device = select_device()
    print(f"Using device: {device}")
    
    # Record start time
    start_time = time.time()
    
    if mode in ['train', 'train_test']:
        train_supervised(relation, data_path)
        train_rl(relation, data_path, max_episodes)
    
    if mode in ['test', 'train_test']:
        test(relation, data_path)
    
    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f}s")

if __name__ == "__main__":
    main()