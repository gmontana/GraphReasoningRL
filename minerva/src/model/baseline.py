import torch
import numpy as np


class ReactiveBaseline:
    """
    Reactive baseline for variance reduction in REINFORCE.
    Updates baseline value using exponential moving average.
    """
    def __init__(self, l=0.0):
        self.l = l  # Learning rate for baseline update
        self.b = 0.0  # Current baseline value
        
    def update(self, reward):
        """Update baseline with new reward value"""
        self.b = self.l * self.b + (1 - self.l) * reward
        
    def get_baseline_value(self):
        """Get current baseline value"""
        return self.b