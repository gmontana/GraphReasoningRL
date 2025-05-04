"""
Utility functions for DeepPath.

This module contains utility functions used throughout the DeepPath package,
including transition storage, path cleaning, and various helper functions.
"""

import numpy as np
import logging
import sys
import os
from collections import namedtuple, Counter
from datetime import datetime

# Constants - these should match your data
STATE_DIM = 200  # 2 * entity embedding dimension
ACTION_SPACE = 10  # Number of relations in the dataset - match TF implementation
MAX_STEPS = 50
MAX_STEPS_TEST = 50
EMBEDDING_DIM = 100  # Entity embedding dimension

# Default data path
DEFAULT_DATA_PATH = '../NELL-995/'

# Define a named tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def distance(e1, e2):
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        e1 (np.ndarray): First embedding
        e2 (np.ndarray): Second embedding
        
    Returns:
        float: Euclidean distance between e1 and e2
    """
    return np.sqrt(np.sum(np.square(e1 - e2)))


def compare(v1, v2):
    """
    Count the number of matching elements between two arrays.
    
    Args:
        v1 (np.ndarray): First array
        v2 (np.ndarray): Second array
        
    Returns:
        int: Number of matching elements
    """
    return sum(v1 == v2)


def path_clean(path):
    """
    Clean a path by removing duplicate entities.
    
    Args:
        path (str): Path string with entities and relations
        
    Returns:
        str: Cleaned path without duplicates
    """
    rel_ents = path.split(' -> ')
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    
    entity_stats = Counter(entities).items()
    duplicate_ents = [item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda x: x[1], reverse=True)
    
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx != max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
    
    return ' -> '.join(rel_ents)


def prob_norm(probs):
    """
    Normalize probabilities to sum to 1.
    
    Args:
        probs (np.ndarray): Array of probabilities
        
    Returns:
        np.ndarray: Normalized probabilities
    """
    return probs / sum(probs)


def get_logger(name, level=logging.INFO, log_dir=None):
    """
    Configure and get a logger with the given name and level.
    
    Args:
        name (str): Logger name
        level (int): Logging level
        log_dir (str, optional): Directory to save log files. If None, logs are only written to stderr.
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_success_rate(successes, total):
    """
    Calculate success rate as a percentage.
    
    Args:
        successes (int): Number of successful episodes
        total (int): Total number of episodes
        
    Returns:
        float: Success rate as a percentage
    """
    return (successes / total) * 100


def select_device():
    """
    Select the appropriate PyTorch device based on availability.
    
    Returns:
        torch.device: Selected device (MPS, CUDA, or CPU)
    """
    import torch
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device