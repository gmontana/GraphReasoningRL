"""
DeepPath: Reinforcement Learning for Knowledge Graph Reasoning

A package for knowledge graph reasoning using reinforcement learning.
"""

__version__ = "0.2.0"

from .models import PolicyNetwork, ValueNetwork, QNetwork
from .agents import PolicyAgent, SupervisedPolicyAgent, train_reinforce
from .environment import Env
from .search import KB, Path, bfs_two_way as BFS, teacher
from .utils import Transition, path_clean, distance, compare, prob_norm, select_device

__all__ = [
    "PolicyNetwork", "ValueNetwork", "QNetwork", 
    "PolicyAgent", "SupervisedPolicyAgent", "train_reinforce",
    "Env", "KB", "Path", "BFS", "teacher",
    "Transition", "path_clean", "distance", "compare", "prob_norm", "select_device"
]