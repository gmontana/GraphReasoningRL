import torch
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from options import read_options
from model.trainer import Trainer
from utils import device_info

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    # Read options
    options = read_options()
    
    # Show device information
    device_info()
    
    # Add file handler for logging
    if 'log_file_name' in options:
        file_handler = logging.FileHandler(options['log_file_name'])
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    # Create trainer
    trainer = Trainer(options)
    
    if options['load_model']:
        # Load model for testing
        logger.info(f"Loading model from {options['model_load_dir']}")
        trainer.load_model(options['model_load_dir'])
        
        # Test on test set
        logger.info("Evaluating on test set...")
        trainer.test_environment = trainer.test_test_environment
        results = trainer.test(beam=True, print_paths=False, save_model=False)
        
        logger.info("Test results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
    else:
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Final test
        logger.info("Final evaluation on test set...")
        trainer.test_environment = trainer.test_test_environment
        results = trainer.test(beam=True, print_paths=False, save_model=False)
        
        logger.info("Final test results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()