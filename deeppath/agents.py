"""
Agent implementations for DeepPath.

This module contains the agent implementations for DeepPath, including the
policy agent that learns to find paths in knowledge graphs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
from itertools import count
import os
import time
from sklearn.metrics.pairwise import cosine_similarity

from .models import PolicyNetwork
from .utils import Transition, path_clean, STATE_DIM, ACTION_SPACE, MAX_STEPS, MAX_STEPS_TEST, EMBEDDING_DIM, select_device
from .search import teacher

# Get the optimal device (MPS for Apple Silicon, CUDA, or CPU)
device = select_device()


class PolicyAgent:
    """
    Policy-based agent for knowledge graph reasoning.
    
    This agent uses a policy network to learn to navigate knowledge graphs.
    """
    def __init__(self, scope='policy_network', learning_rate=0.001):
        """
        Initialize the policy agent.
        
        Args:
            scope (str): Name scope for the agent
            learning_rate (float): Learning rate for optimization
        """
        self.model = PolicyNetwork(STATE_DIM, ACTION_SPACE).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
    def predict(self, state):
        """
        Predict action probabilities for a given state.
        
        Args:
            state (np.ndarray): State representation
            
        Returns:
            np.ndarray: Action probabilities
        """
        # Convert numpy array to torch tensor
        state_tensor = torch.FloatTensor(state).to(device)
        
        # Make sure we have a batch dimension
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        self.model.eval()
        with torch.no_grad():
            action_prob = self.model(state_tensor)
        return action_prob.cpu().numpy()
        
    def choose_action(self, state):
        """
        Choose an action based on the current state.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Selected action
        """
        # Get action probabilities
        action_probs = self.predict(state)
        
        # Sample an action from the probability distribution
        action = np.random.choice(np.arange(ACTION_SPACE), p=np.squeeze(action_probs))
        
        return action
        
    def update(self, state, target, action):
        """
        Update the policy network with a batch of experiences.
        
        Args:
            state (np.ndarray): Batch of states
            target (float): Target value
            action (list): Batch of actions
            
        Returns:
            float: Loss value
        """
        state_tensor = torch.FloatTensor(state).to(device)
        target_tensor = torch.FloatTensor([target]).to(device)
        action_tensor = torch.LongTensor(action).to(device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        action_probs = self.model(state_tensor)
        
        # Create one-hot mask for actions
        action_mask = torch.zeros_like(action_probs).to(device)
        for i in range(len(action)):
            action_mask[i, action[i]] = 1
        
        # Select the probabilities of chosen actions
        picked_action_probs = torch.sum(action_probs * action_mask, dim=1)
        
        # Calculate negative log likelihood loss with regularization
        loss = -torch.sum(torch.log(picked_action_probs) * target_tensor)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        # Silent save - no print statements
    
    def load(self, path):
        """
        Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        self.model.load_state_dict(torch.load(path, map_location=device))


class SupervisedPolicyAgent:
    """
    Agent that learns from supervised examples.
    
    This agent is trained using supervised learning with expert demonstrations.
    """
    def __init__(self, learning_rate=0.001):
        """
        Initialize the supervised policy agent.
        
        Args:
            learning_rate (float): Learning rate for optimization
        """
        self.model = PolicyNetwork(STATE_DIM, ACTION_SPACE).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
    def predict(self, state):
        """
        Predict action probabilities for a given state.
        
        Args:
            state (np.ndarray): State representation
            
        Returns:
            np.ndarray: Action probabilities
        """
        # Convert numpy array to torch tensor
        state_tensor = torch.FloatTensor(state).to(device)
        
        # Make sure we have a batch dimension
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        self.model.eval()
        with torch.no_grad():
            action_prob = self.model(state_tensor)
        return action_prob.cpu().numpy()
        
    def update(self, state, action):
        """
        Update the policy network with a batch of supervised examples.
        
        Args:
            state (np.ndarray): Batch of states
            action (list): Batch of actions
            
        Returns:
            float: Loss value
        """
        state_tensor = torch.FloatTensor(state).to(device)
        action_tensor = torch.LongTensor(action).to(device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        action_probs = self.model(state_tensor)
        
        # Create a mask for the chosen actions
        action_mask = torch.zeros_like(action_probs).to(device)
        for i in range(len(action)):
            action_mask[i, action[i]] = 1
        
        # Select the probabilities of chosen actions
        picked_action_probs = torch.sum(action_probs * action_mask, dim=1)
        
        # Calculate negative log likelihood loss
        loss = -torch.sum(torch.log(picked_action_probs))
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        # Silent save - no print statements
    
    def load(self, path):
        """
        Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        self.model.load_state_dict(torch.load(path, map_location=device))


def train_reinforce(agent, train_data, env_factory, graph_path, num_episodes):
    """
    Train an agent using the REINFORCE algorithm.
    
    Args:
        agent (PolicyAgent): The agent to train
        train_data (list): List of training examples
        env_factory (callable): Function to create environments
        graph_path (str): Path to the knowledge graph
        num_episodes (int): Number of episodes to train for
        
    Returns:
        tuple: (success_count, path_stats)
    """
    success = 0
    path_found_entity = []
    path_relation_found = []
    
    for i_episode in range(num_episodes):
        start = time.time()
        print('Episode %d' % i_episode)
        print('Training sample: ', train_data[i_episode][:-1])
        
        env = env_factory(train_data[i_episode])
        
        sample = train_data[i_episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
        
        episode = []
        state_batch_negative = []
        action_batch_negative = []
        
        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = agent.predict(state_vec)
            action_chosen = np.random.choice(np.arange(ACTION_SPACE), p=np.squeeze(action_probs))
            reward, new_state, done = env.interact(state_idx, action_chosen)
            
            if reward == -1:  # the action fails for this step
                state_batch_negative.append(state_vec)
                action_batch_negative.append(action_chosen)
            
            # Handle case where new_state is None (reached target)
            if new_state is None:
                new_state_vec = None
            else:
                new_state_vec = env.idx_state(new_state)
                
            episode.append(Transition(state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))
            
            if done or t == MAX_STEPS:
                break
            
            state_idx = new_state
        
        # Discourage the agent when it chooses an invalid step
        if len(state_batch_negative) != 0:
            print('Penalty to invalid steps:', len(state_batch_negative))
            agent.update(np.reshape(state_batch_negative, (-1, STATE_DIM)), -0.05, action_batch_negative)
        
        print('----- FINAL PATH -----')
        print('\t'.join(env.path))
        print('PATH LENGTH', len(env.path))
        print('----- FINAL PATH -----')
        
        # If the agent succeeds, do one optimization
        if done == 1:
            print('Success')
            
            path_found_entity.append(path_clean(' -> '.join(env.path)))
            
            success += 1
            path_length = len(env.path)
            length_reward = 1/path_length
            global_reward = 1
            
            total_reward = 0.1*global_reward + 0.9*length_reward
            state_batch = []
            action_batch = []
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            agent.update(np.reshape(state_batch, (-1, STATE_DIM)), total_reward, action_batch)
        else:
            global_reward = -0.05
            
            state_batch = []
            action_batch = []
            total_reward = global_reward
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            agent.update(np.reshape(state_batch, (-1, STATE_DIM)), total_reward, action_batch)
            
            print('Failed, Do one teacher guideline')
            try:
                good_episodes = teacher(sample[0], sample[1], 1, env, graph_path)
                for item in good_episodes:
                    teacher_state_batch = []
                    teacher_action_batch = []
                    total_reward = 0.0*1 + 1*1/len(item)
                    for t, transition in enumerate(item):
                        teacher_state_batch.append(transition.state)
                        teacher_action_batch.append(transition.action)
                    agent.update(np.squeeze(teacher_state_batch), 1, teacher_action_batch)
            
            except Exception as e:
                print('Teacher guideline failed')
        
        print('Episode time: ', time.time() - start)
        print('\n')
    
    print('Success percentage:', success/num_episodes)
    
    # Collect path statistics
    for path in path_found_entity:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx%2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))
    
    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x:x[1], reverse=True)
    
    return success, relation_path_stats