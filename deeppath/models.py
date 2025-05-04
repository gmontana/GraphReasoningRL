"""
Neural network models for DeepPath.

This module contains all the neural network architectures used in DeepPath,
including policy networks, value networks, and Q-networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Policy Network that predicts action probabilities given a state.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initialize the policy network.
        
        Args:
            state_dim (int): Dimensionality of the state space
            action_dim (int): Dimensionality of the action space
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, action_dim)
        
        # Weight initialization similar to Xavier init in TF
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, action_dim)
                        representing action probabilities
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class ValueNetwork(nn.Module):
    """
    Value Network that estimates the value of a state.
    """
    def __init__(self, state_dim):
        """
        Initialize the value network.
        
        Args:
            state_dim (int): Dimensionality of the state space
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Output tensor with shape (batch_size,)
                        representing state values
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze()


class QNetwork(nn.Module):
    """
    Q-Network that estimates action values given a state.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initialize the Q-network.
        
        Args:
            state_dim (int): Dimensionality of the state space
            action_dim (int): Dimensionality of the action space
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, action_dim)
                        representing action values
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_weights(self):
        """
        Get model weights for compatibility with original implementation.
        
        Returns:
            list: List of network parameters and forward function
        """
        return [
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias,
            self.forward
        ]