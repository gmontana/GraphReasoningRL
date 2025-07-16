#!/usr/bin/env python
"""
Test script to verify device detection and compatibility
"""
import torch
from utils import get_device, device_info


def test_device_detection():
    """Test device detection with different preferences"""
    print("=" * 60)
    print("MINERVA PyTorch Device Detection Test")
    print("=" * 60)
    
    # Show device information
    device_info()
    print()
    
    # Test auto-detection
    print("Testing auto-detection:")
    device = get_device(None)
    print(f"  Auto-detected device: {device}")
    print()
    
    # Test each device preference
    for pref in ['cuda', 'mps', 'cpu']:
        print(f"Testing preference '{pref}':")
        device = get_device(pref)
        print(f"  Selected device: {device}")
        
        # Test tensor creation on device
        try:
            test_tensor = torch.randn(10, 10, device=device)
            print(f"  ✓ Successfully created tensor on {device}")
            
            # Test basic operation
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"  ✓ Successfully performed matrix multiplication on {device}")
        except Exception as e:
            print(f"  ✗ Error on {device}: {e}")
        print()
    
    # Test LSTM on different devices
    print("Testing LSTM on available devices:")
    for device_type in ['cpu', 'mps'] + (['cuda'] if torch.cuda.is_available() else []):
        try:
            device = torch.device(device_type)
            if device_type == 'mps' and not torch.backends.mps.is_available():
                continue
            if device_type == 'cuda' and not torch.cuda.is_available():
                continue
                
            lstm = torch.nn.LSTM(100, 200, num_layers=2, batch_first=True).to(device)
            input_tensor = torch.randn(32, 10, 100, device=device)
            output, (h, c) = lstm(input_tensor)
            print(f"  ✓ LSTM works on {device_type}")
        except Exception as e:
            print(f"  ✗ LSTM failed on {device_type}: {e}")
    
    print("\n" + "=" * 60)
    print("Device test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_device_detection()