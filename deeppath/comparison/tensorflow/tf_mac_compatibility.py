"""
Compatibility layer for running TensorFlow on Mac with Apple Silicon
"""
import tensorflow as tf
import os
import sys

# Check if we're running on Mac
import platform
is_mac = platform.system() == 'Darwin'

# Print TensorFlow version information
print("TensorFlow Version:", tf.__version__)
print("Python Version:", sys.version)
print("Running on Mac:", is_mac)

# Add compatibility code for TensorFlow 1.x API if needed
if not hasattr(tf, 'placeholder'):
    print("Adding TensorFlow 1.x API compatibility...")
    tf.placeholder = tf.compat.v1.placeholder
    tf.Session = tf.compat.v1.Session
    tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
    tf.train = tf.compat.v1.train

# Execute the requested script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tf_mac_compatibility.py <script_to_run.py> [script args...]")
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
