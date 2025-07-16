import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device_preference=None):
    """
    Get the best available device, with preference order:
    1. User specified device (if valid)
    2. CUDA (if available)
    3. MPS (if available, for Apple Silicon)
    4. CPU (fallback)
    
    Args:
        device_preference: str or None - 'cuda', 'mps', 'cpu', or None for auto-detect
        
    Returns:
        torch.device: The selected device
    """
    if device_preference:
        device_preference = device_preference.lower()
        
        if device_preference == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                return device
            else:
                logger.warning("CUDA requested but not available, falling back to auto-detection")
                
        elif device_preference == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using MPS device (Apple Silicon)")
                return device
            else:
                logger.warning("MPS requested but not available, falling back to auto-detection")
                
        elif device_preference == 'cpu':
            device = torch.device('cpu')
            logger.info("Using CPU device (user specified)")
            return device
        else:
            logger.warning(f"Unknown device preference: {device_preference}, using auto-detection")
    
    # Auto-detection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
        
    return device


def device_info():
    """
    Print information about available devices
    """
    print("PyTorch Device Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("MPS device: Apple Silicon GPU")