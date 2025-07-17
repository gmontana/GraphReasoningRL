import torch
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from options import read_options
from model.trainer import Trainer
from model.nell_eval import nell_eval
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
        
        # Set output directory for file saving if not already set
        if 'output_dir' not in options:
            options['output_dir'] = os.path.dirname(options['model_load_dir'])
        trainer.output_dir = options['output_dir']
        
        # Test on test set
        logger.info("Evaluating on test set...")
        trainer.test_environment = trainer.test_test_environment
        print_paths = options['nell_evaluation'] == 1
        results = trainer.test(beam=True, print_paths=print_paths, save_model=False)
        
        logger.info("Test results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # NELL evaluation if enabled
        if options['nell_evaluation'] == 1:
            logger.info("Performing NELL evaluation...")
            # For loaded models, use the model load directory's parent
            if 'output_dir' in options:
                output_dir = options['output_dir']
            else:
                output_dir = os.path.dirname(options['model_load_dir'])
            
            answers_file = os.path.join(output_dir, 'test_beam', 'pathsanswers')
            test_pairs_file = os.path.join(options['data_input_dir'], 'sort_test.pairs')
            
            if os.path.exists(answers_file) and os.path.exists(test_pairs_file):
                nell_eval(answers_file, test_pairs_file)
            else:
                logger.warning(f"NELL evaluation files not found: {answers_file}, {test_pairs_file}")
    else:
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Final test
        logger.info("Final evaluation on test set...")
        trainer.test_environment = trainer.test_test_environment
        print_paths = options['nell_evaluation'] == 1
        results = trainer.test(beam=True, print_paths=print_paths, save_model=False)
        
        logger.info("Final test results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # NELL evaluation if enabled
        if options['nell_evaluation'] == 1:
            logger.info("Performing NELL evaluation...")
            answers_file = os.path.join(options['output_dir'], 'test_beam', 'pathsanswers')
            test_pairs_file = os.path.join(options['data_input_dir'], 'sort_test.pairs')
            
            if os.path.exists(answers_file) and os.path.exists(test_pairs_file):
                nell_eval(answers_file, test_pairs_file)
            else:
                logger.warning(f"NELL evaluation files not found: {answers_file}, {test_pairs_file}")


if __name__ == '__main__':
    main()