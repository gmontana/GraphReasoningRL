#!/usr/bin/env python
"""
DeepPath evaluation module - Implements logic evaluation similar to
the original TensorFlow implementation's evaluate.py script.
"""

import os
import numpy as np
from collections import defaultdict
from .search import KB, bfs_two_way
from .utils import select_device

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

class EvaluationDataset(Dataset):
    """Dataset for evaluation paths and labels."""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EvaluationModel(nn.Module):
    """Simple model for relation path evaluation."""
    def __init__(self, input_dim):
        super(EvaluationModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def get_features(feature_path, stats_path, relation_path):
    """Extract and process features from paths."""
    # Read path statistics
    stats = {}
    with open(stats_path) as f:
        path_freq = f.readlines()
    for line in path_freq:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            path, num = parts
            stats[path] = int(num)
    
    # Load relation to ID mapping
    relation2id = {}
    with open(relation_path) as f:
        content = f.readlines()
    for line in content:
        parts = line.strip().split()
        if len(parts) == 2:
            relation2id[parts[0]] = int(parts[1])
    
    # Process useful paths
    useful_paths = []
    named_paths = []
    with open(feature_path) as f:
        paths = f.readlines()
    
    print(f"Total paths: {len(paths)}")
    
    for line in paths:
        path = line.strip()
        length = len(path.split(' -> '))
        
        if length <= 10:  # Filter by length just like original
            pathIndex = []
            pathName = []
            relations = path.split(' -> ')
            
            for rel in relations:
                pathName.append(rel)
                if rel in relation2id:
                    rel_id = relation2id[rel]
                    pathIndex.append(rel_id)
                else:
                    # Skip paths with unknown relations
                    break
            else:  # Only if the loop completed normally
                useful_paths.append(pathIndex)
                named_paths.append(pathName)
    
    print(f'Paths used: {len(useful_paths)}')
    return useful_paths, named_paths

def train_pytorch(kb, kb_inv, named_paths, train_path, device='cpu'):
    """Train evaluation model using PyTorch."""
    # Read training data
    with open(train_path) as f:
        train_data = f.readlines()
    
    train_pairs = []
    train_labels = []
    for line in train_data:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            e1 = parts[0].replace('thing$', '')
            e2_part = parts[1]
            if ':' in e2_part:
                e2 = e2_part.split(':')[0].replace('thing$', '')
            else:
                e2 = e2_part.replace('thing$', '')
                
            if (e1 not in kb.entities) or (e2 not in kb.entities):
                continue
            train_pairs.append((e1, e2))
            label = 1 if line.strip()[-1] == '+' else 0
            train_labels.append(label)
    
    # Extract features for training
    training_features = []
    for sample in train_pairs:
        feature = []
        for path in named_paths:
            feature.append(int(bfs_two_way(sample[0], sample[1], path, kb, kb_inv)))
        training_features.append(feature)
    
    # Create model, dataset and dataloader
    input_dim = len(named_paths)
    model = EvaluationModel(input_dim).to(device)
    optimizer = optim.RMSprop(model.parameters())
    criterion = nn.BCELoss()
    
    # Create dataset and dataloader
    dataset = EvaluationDataset(training_features, train_labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Train model
    epochs = 300
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model

def evaluate_logic(relation, data_path='./NELL-995/tasks/'):
    """Evaluate logic paths using the trained model."""
    device = select_device()
    print(f"Using device: {device}")
    
    relation_path = data_path + relation
    feature_path = os.path.join(relation_path, 'path_to_use.txt')
    feature_stats = os.path.join(relation_path, 'path_stats.txt')
    relation_id_path = os.path.join(os.path.dirname(data_path), 'relation2id.txt')
    train_path = os.path.join(relation_path, 'train.pairs')
    test_path = os.path.join(relation_path, 'sort_test.pairs')
    graph_path = os.path.join(relation_path, 'graph.txt')
    
    # Check file existence
    for path in [feature_path, feature_stats, relation_id_path, graph_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
    
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found, using train_pos instead")
        train_path = os.path.join(relation_path, 'train_pos')
    
    if not os.path.exists(test_path):
        print(f"Warning: {test_path} not found, using train_pos for testing")
        test_path = os.path.join(relation_path, 'train_pos')
    
    # Create KBs
    kb = KB()
    kb_inv = KB()
    
    with open(graph_path) as f:
        kb_lines = f.readlines()
    
    print("Loading knowledge graph...")
    for line in kb_lines:
        parts = line.strip().split()
        if len(parts) == 3:
            e1, rel, e2 = parts
            kb.addRelation(e1, rel, e2)
            kb_inv.addRelation(e2, rel, e1)
    
    # Get features
    _, named_paths = get_features(feature_path, feature_stats, relation_id_path)
    
    # Train model
    print("Training evaluation model...")
    model = train_pytorch(kb, kb_inv, named_paths, train_path, device)
    
    # Test model
    with open(test_path) as f:
        test_data = f.readlines()
    
    test_pairs = []
    test_labels = []
    for line in test_data:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            e1 = parts[0].replace('thing$', '')
            e2_part = parts[1]
            if ':' in e2_part:
                e2 = e2_part.split(':')[0].replace('thing$', '')
            else:
                e2 = e2_part.replace('thing$', '')
                
            if (e1 not in kb.entities) or (e2 not in kb.entities):
                continue
            test_pairs.append((e1, e2))
            label = 1 if line.strip()[-1] == '+' else 0
            test_labels.append(label)
    
    # Calculate Mean Average Precision
    aps = []
    query = test_pairs[0][0] if test_pairs else None
    y_true = []
    y_score = []
    score_all = []
    
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(test_pairs):
            if sample[0] == query:
                features = []
                for path in named_paths:
                    features.append(int(bfs_two_way(sample[0], sample[1], path, kb, kb_inv)))
                
                features_tensor = torch.FloatTensor([features]).to(device)
                score = model(features_tensor).item()
                
                score_all.append(score)
                y_score.append(score)
                y_true.append(test_labels[idx])
            else:
                # New query, calculate AP for previous query
                query = sample[0]
                count = list(zip(y_score, y_true))
                count.sort(key=lambda x: x[0], reverse=True)
                ranks = []
                correct = 0
                for idx_, item in enumerate(count):
                    if item[1] == 1:
                        correct += 1
                        ranks.append(correct / (1.0 + idx_))
                if ranks:
                    aps.append(np.mean(ranks))
                else:
                    aps.append(0)
                
                # Reset for new query
                y_true = []
                y_score = []
                
                # Process new sample
                features = []
                for path in named_paths:
                    features.append(int(bfs_two_way(sample[0], sample[1], path, kb, kb_inv)))
                
                features_tensor = torch.FloatTensor([features]).to(device)
                score = model(features_tensor).item()
                
                score_all.append(score)
                y_score.append(score)
                y_true.append(test_labels[idx])
    
    # Calculate AP for the last query
    if y_score:
        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)
        ranks = []
        correct = 0
        for idx_, item in enumerate(count):
            if item[1] == 1:
                correct += 1
                ranks.append(correct / (1.0 + idx_))
        if ranks:
            aps.append(np.mean(ranks))
        else:
            aps.append(0)
    
    mean_ap = np.mean(aps) if aps else 0
    print(f'PyTorch MAP: {mean_ap:.4f}')
    
    # Write evaluation result to file
    eval_result_path = os.path.join(relation_path, 'path_evaluation.txt')
    with open(eval_result_path, 'w') as f:
        f.write(f'MAP: {mean_ap:.4f}\n')
        
    return mean_ap

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        relation = sys.argv[1]
        evaluate_logic(relation)
    else:
        print("Usage: python evaluate.py <relation>")
        print("Example: python evaluate.py athletePlaysForTeam")