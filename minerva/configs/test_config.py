#!/usr/bin/env python3
"""
Test script to verify configuration loading works correctly.
"""

import sys
import os
sys.path.append('../src')

from options import read_options

def test_config_loading():
    """Test loading configuration from JSON files."""
    print("Testing configuration loading...")
    
    # Test countries_s1 config
    config_file = "countries_s1.json"
    if os.path.exists(config_file):
        print(f"\n✓ Testing {config_file}")
        
        # Temporarily modify sys.argv to test JSON loading
        original_argv = sys.argv[:]
        sys.argv = ['test_config.py', config_file]
        
        try:
            options = read_options()
            print(f"  - Data directory: {options['data_input_dir']}")
            print(f"  - Path length: {options['path_length']}")
            print(f"  - Hidden size: {options['hidden_size']}")
            print(f"  - Batch size: {options['batch_size']}")
            print(f"  - Total iterations: {options['total_iterations']}")
            print(f"  - NELL evaluation: {options['nell_evaluation']}")
            print("  ✓ Configuration loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading configuration: {e}")
        finally:
            sys.argv = original_argv
    else:
        print(f"✗ Configuration file {config_file} not found")
    
    print("\nTest completed!")


if __name__ == '__main__':
    test_config_loading()