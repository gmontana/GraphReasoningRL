#!/usr/bin/env python
"""
Compatibility layer for running DeepPath TensorFlow 1.x code in TensorFlow 2.x
"""

import tensorflow as tf
import os
import sys

# Enable eager execution for TensorFlow 2.x
tf.compat.v1.enable_eager_execution()

# Create TensorFlow 1.x compatibility layer
tf.compat.v1.disable_v2_behavior()

# Add missing contrib module
class LayersModule:
    @staticmethod
    def xavier_initializer():
        # Simple implementation that doesn't require keras
        return lambda shape, dtype=None: tf.random.truncated_normal(shape, stddev=0.001, dtype=dtype)
    
    @staticmethod
    def l2_regularizer(scale):
        # Simple implementation of l2 regularizer
        return lambda weights: tf.multiply(scale, tf.reduce_sum(tf.square(weights)))

class ContribModule:
    def __init__(self):
        self.layers = LayersModule()

# Add contrib module to tf.compat.v1
tf.compat.v1.contrib = ContribModule()

# Set v1 as the default for imports
# This allows the original code to import tensorflow as tf and use v1 operations
sys.modules['tensorflow'] = tf.compat.v1

# Print TensorFlow version information
print("TensorFlow Version:", tf.__version__)
print("Running in v1 compatibility mode with eager execution")
print("Python Version:", sys.version)

# Execute the requested script with the compatibility layer
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tf2_compatibility.py <script_to_run.py> [script args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    
    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found")
        sys.exit(1)
    
    # Set script arguments
    sys.argv = [script_path] + script_args
    
    # Execute the script
    print(f"Running {script_path} with args: {script_args}")
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, globals(), globals())