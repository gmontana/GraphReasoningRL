#!/usr/bin/env python3
"""
NELL dataset preprocessing script for MINERVA.

This script processes NELL dataset files into the format expected by MINERVA.
It handles the specific format of NELL data and creates the necessary files.
"""

import json
import csv
import os
import argparse
import numpy as np
from pathlib import Path


def preprocess_nell_dataset(task_dir, task_name, output_dir):
    """
    Preprocess NELL dataset from raw files to MINERVA format.
    
    Args:
        task_dir: Directory containing raw NELL task files
        task_name: Name of the NELL task/relation
        output_dir: Output directory for processed files
        
    Returns:
        tuple: (entity_vocab, relation_vocab) dictionaries
    """
    # Initialize vocabularies with special tokens
    entity_vocab = {}
    relation_vocab = {}
    
    entity_vocab['PAD'] = len(entity_vocab)
    entity_vocab['UNK'] = len(entity_vocab)
    relation_vocab['PAD'] = len(relation_vocab)
    relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
    relation_vocab['NO_OP'] = len(relation_vocab)
    relation_vocab['UNK'] = len(relation_vocab)
    
    entity_counter = len(entity_vocab)
    relation_counter = len(relation_vocab)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    vocab_dir = os.path.join(output_dir, 'vocab')
    os.makedirs(vocab_dir, exist_ok=True)
    
    target_relation = task_name
    
    print(f"Processing NELL task: {task_name}")
    print(f"Input directory: {task_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process train_pos file to create train.txt and dev.txt
    train_pos_file = os.path.join(task_dir, 'train_pos')
    if not os.path.exists(train_pos_file):
        raise FileNotFoundError(f"train_pos file not found: {train_pos_file}")
    
    print("Processing train_pos file...")
    with open(os.path.join(output_dir, 'train.txt'), 'w') as train_file:
        with open(os.path.join(output_dir, 'dev.txt'), 'w') as dev_file:
            with open(train_pos_file, 'r') as raw_file:
                csv_file = csv.reader(raw_file, delimiter='\t')
                for line in csv_file:
                    if len(line) != 3:
                        print(f"Warning: Invalid line in train_pos: {line}")
                        continue
                        
                    e1, e2, r = line
                    target_relation = r  # Store the actual relation name
                    
                    # Add to vocabularies
                    if e1 not in entity_vocab:
                        entity_vocab[e1] = entity_counter
                        entity_counter += 1
                    if e2 not in entity_vocab:
                        entity_vocab[e2] = entity_counter
                        entity_counter += 1
                    if r not in relation_vocab:
                        relation_vocab[r] = relation_counter
                        relation_counter += 1
                    
                    # Write to train file
                    train_file.write(f"{e1}\\t{r}\\t{e2}\\n")
                    
                    # Randomly add to dev file (80% probability)
                    if np.random.random() > 0.2:
                        dev_file.write(f"{e1}\\t{r}\\t{e2}\\n")
    
    # Process graph.txt file
    graph_file = os.path.join(task_dir, 'graph.txt')
    if os.path.exists(graph_file):
        print("Processing graph.txt file...")
        with open(os.path.join(output_dir, 'graph.txt'), 'w') as out_file:
            # First, process the background graph
            with open(graph_file, 'r') as raw_file:
                csv_file = csv.reader(raw_file, delimiter='\t')
                for line in csv_file:
                    if len(line) != 3:
                        print(f"Warning: Invalid line in graph.txt: {line}")
                        continue
                        
                    e1, r, e2 = line
                    
                    # Add to vocabularies
                    if e1 not in entity_vocab:
                        entity_vocab[e1] = entity_counter
                        entity_counter += 1
                    if e2 not in entity_vocab:
                        entity_vocab[e2] = entity_counter
                        entity_counter += 1
                    if r not in relation_vocab:
                        relation_vocab[r] = relation_counter
                        relation_counter += 1
                    
                    out_file.write(f"{e1}\\t{r}\\t{e2}\\n")
            
            # Then, add the training positive examples to the graph
            with open(train_pos_file, 'r') as raw_file:
                csv_file = csv.reader(raw_file, delimiter='\t')
                for line in csv_file:
                    if len(line) != 3:
                        continue
                        
                    e1, e2, r = line
                    
                    # Add to vocabularies (might be duplicates, but that's OK)
                    if e1 not in entity_vocab:
                        entity_vocab[e1] = entity_counter
                        entity_counter += 1
                    if e2 not in entity_vocab:
                        entity_vocab[e2] = entity_counter
                        entity_counter += 1
                    if r not in relation_vocab:
                        relation_vocab[r] = relation_counter
                        relation_counter += 1
                    
                    out_file.write(f"{e1}\\t{r}\\t{e2}\\n")
    
    # Process test.pairs file to create test.txt
    test_pairs_file = os.path.join(task_dir, 'test.pairs')
    if os.path.exists(test_pairs_file):
        print("Processing test.pairs file...")
        with open(os.path.join(output_dir, 'test.txt'), 'w') as out_file:
            with open(test_pairs_file, 'r') as raw_file:
                for line in raw_file:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Parse NELL test pair format: "thing$entity1,thing$entity2	+"
                    if line.endswith('+'):
                        # Extract entity pair
                        entity_part = line[:-2].strip()  # Remove '\t+'
                        
                        if ',' in entity_part:
                            e1_part, e2_part = entity_part.split(',', 1)
                            
                            # Remove 'thing$' prefix if present
                            e1 = e1_part.replace('thing$', '') if 'thing$' in e1_part else e1_part
                            e2 = e2_part.replace('thing$', '') if 'thing$' in e2_part else e2_part
                            
                            # Add to vocabularies
                            if e1 not in entity_vocab:
                                entity_vocab[e1] = entity_counter
                                entity_counter += 1
                            if e2 not in entity_vocab:
                                entity_vocab[e2] = entity_counter
                                entity_counter += 1
                            if target_relation not in relation_vocab:
                                relation_vocab[target_relation] = relation_counter
                                relation_counter += 1
                            
                            out_file.write(f"{e1}\\t{target_relation}\\t{e2}\\n")
    
    # Create sort_test.pairs file (copy of original test.pairs for NELL evaluation)
    sort_test_file = os.path.join(output_dir, 'sort_test.pairs')
    if os.path.exists(test_pairs_file):
        print("Creating sort_test.pairs file...")
        import shutil
        shutil.copy2(test_pairs_file, sort_test_file)
    
    return entity_vocab, relation_vocab, vocab_dir


def save_vocabularies(entity_vocab, relation_vocab, vocab_dir):
    """Save vocabularies to JSON files."""
    entity_vocab_file = os.path.join(vocab_dir, 'entity_vocab.json')
    relation_vocab_file = os.path.join(vocab_dir, 'relation_vocab.json')
    
    with open(entity_vocab_file, 'w', encoding='utf-8') as fout:
        json.dump(entity_vocab, fout, indent=2)
    
    with open(relation_vocab_file, 'w', encoding='utf-8') as fout:
        json.dump(relation_vocab, fout, indent=2)
    
    print(f"Saved entity vocabulary to {entity_vocab_file}")
    print(f"Saved relation vocabulary to {relation_vocab_file}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess NELL datasets for MINERVA')
    parser.add_argument("--task_dir", required=True, type=str,
                       help="Directory containing raw NELL task files")
    parser.add_argument("--task_name", required=True, type=str,
                       help="Name of the NELL task/relation")
    parser.add_argument("--output_dir", required=True, type=str,
                       help="Output directory for processed files")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.task_dir):
        raise ValueError(f"Task directory {args.task_dir} does not exist")
    
    # Check for required files
    required_files = ['train_pos']
    for filename in required_files:
        filepath = os.path.join(args.task_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")
    
    # Process dataset
    entity_vocab, relation_vocab, vocab_dir = preprocess_nell_dataset(
        args.task_dir, args.task_name, args.output_dir
    )
    
    # Save vocabularies
    save_vocabularies(entity_vocab, relation_vocab, vocab_dir)
    
    # Print statistics
    print(f"\\nPreprocessing completed!")
    print(f"Vocabulary Statistics:")
    print(f"  Entities: {len(entity_vocab)}")
    print(f"  Relations: {len(relation_vocab)}")
    
    # List created files
    created_files = []
    for filename in ['train.txt', 'dev.txt', 'test.txt', 'graph.txt', 'sort_test.pairs']:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            created_files.append(filename)
    
    print(f"\\nCreated files in {args.output_dir}:")
    for filename in created_files:
        print(f"  {filename}")
    
    print(f"\\nVocabulary files in {vocab_dir}:")
    print(f"  entity_vocab.json")
    print(f"  relation_vocab.json")


if __name__ == '__main__':
    main()