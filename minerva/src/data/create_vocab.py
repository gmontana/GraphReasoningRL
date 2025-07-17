#!/usr/bin/env python3
"""
Generic vocabulary creation script for MINERVA datasets.

This script creates entity and relation vocabularies from knowledge graph files.
It processes train.txt, dev.txt, test.txt, graph.txt, and full_graph.txt files.
"""

import json
import csv
import argparse
import os
from pathlib import Path


def create_vocabularies(data_dir, vocab_dir, files=None):
    """
    Create entity and relation vocabularies from dataset files.
    
    Args:
        data_dir: Directory containing the dataset files
        vocab_dir: Directory to save the vocabulary files
        files: List of files to process (default: standard files)
    
    Returns:
        tuple: (entity_vocab, relation_vocab) dictionaries
    """
    if files is None:
        files = ['train.txt', 'dev.txt', 'test.txt', 'graph.txt', 'full_graph.txt']
    
    # Initialize vocabularies with special tokens
    entity_vocab = {}
    relation_vocab = {}
    
    # Add special tokens - maintain compatibility with original
    entity_vocab['PAD'] = len(entity_vocab)
    entity_vocab['UNK'] = len(entity_vocab)
    relation_vocab['PAD'] = len(relation_vocab)
    relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
    relation_vocab['NO_OP'] = len(relation_vocab)
    relation_vocab['UNK'] = len(relation_vocab)
    
    entity_counter = len(entity_vocab)
    relation_counter = len(relation_vocab)
    
    # Process each file
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found, skipping...")
            continue
            
        print(f"Processing {filename}...")
        with open(filepath, 'r', encoding='utf-8') as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            for line_num, line in enumerate(csv_file):
                try:
                    if len(line) != 3:
                        print(f"Warning: Line {line_num + 1} in {filename} has {len(line)} columns, expected 3")
                        continue
                        
                    e1, r, e2 = line
                    
                    # Add entities to vocabulary
                    if e1 not in entity_vocab:
                        entity_vocab[e1] = entity_counter
                        entity_counter += 1
                    if e2 not in entity_vocab:
                        entity_vocab[e2] = entity_counter
                        entity_counter += 1
                    
                    # Add relation to vocabulary
                    if r not in relation_vocab:
                        relation_vocab[r] = relation_counter
                        relation_counter += 1
                        
                except Exception as e:
                    print(f"Error processing line {line_num + 1} in {filename}: {e}")
                    continue
    
    return entity_vocab, relation_vocab


def save_vocabularies(entity_vocab, relation_vocab, vocab_dir):
    """Save vocabularies to JSON files."""
    os.makedirs(vocab_dir, exist_ok=True)
    
    entity_vocab_file = os.path.join(vocab_dir, 'entity_vocab.json')
    relation_vocab_file = os.path.join(vocab_dir, 'relation_vocab.json')
    
    with open(entity_vocab_file, 'w', encoding='utf-8') as fout:
        json.dump(entity_vocab, fout, indent=2)
    
    with open(relation_vocab_file, 'w', encoding='utf-8') as fout:
        json.dump(relation_vocab, fout, indent=2)
    
    print(f"Saved entity vocabulary to {entity_vocab_file}")
    print(f"Saved relation vocabulary to {relation_vocab_file}")


def main():
    parser = argparse.ArgumentParser(description='Create vocabularies for MINERVA datasets')
    parser.add_argument("--data_input_dir", required=True, type=str, 
                       help="Input data directory containing dataset files")
    parser.add_argument("--vocab_dir", required=True, type=str, 
                       help="Output directory for vocabulary files")
    parser.add_argument("--files", nargs='+', default=None,
                       help="List of files to process (default: train.txt, dev.txt, test.txt, graph.txt, full_graph.txt)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.data_input_dir):
        raise ValueError(f"Data input directory {args.data_input_dir} does not exist")
    
    print(f"Creating vocabularies from {args.data_input_dir}")
    print(f"Output directory: {args.vocab_dir}")
    
    # Create vocabularies
    entity_vocab, relation_vocab = create_vocabularies(
        args.data_input_dir, args.vocab_dir, args.files
    )
    
    # Save vocabularies
    save_vocabularies(entity_vocab, relation_vocab, args.vocab_dir)
    
    # Print statistics
    print(f"\nVocabulary Statistics:")
    print(f"  Entities: {len(entity_vocab)}")
    print(f"  Relations: {len(relation_vocab)}")
    
    print(f"\nEntity vocabulary sample:")
    for i, (entity, idx) in enumerate(list(entity_vocab.items())[:10]):
        print(f"  {entity}: {idx}")
    if len(entity_vocab) > 10:
        print(f"  ... and {len(entity_vocab) - 10} more")
    
    print(f"\nRelation vocabulary sample:")
    for i, (relation, idx) in enumerate(list(relation_vocab.items())[:10]):
        print(f"  {relation}: {idx}")
    if len(relation_vocab) > 10:
        print(f"  ... and {len(relation_vocab) - 10} more")


if __name__ == '__main__':
    main()