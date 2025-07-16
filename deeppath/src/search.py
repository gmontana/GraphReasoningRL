#!/usr/bin/env python
"""
DeepPath search module - Implements search algorithms for path finding in knowledge graphs.
"""

from collections import defaultdict
import queue
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Any

@dataclass
class Path:
    """Represents a relation path in the knowledge graph."""
    relation: str
    connected_entity: str

class KB:
    """Knowledge base that stores relation triples."""
    def __init__(self):
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.entity_to_paths: Dict[str, List[Path]] = defaultdict(list)
    
    def addRelation(self, entity1: str, relation: str, entity2: str) -> None:
        """Add a relation triple to the KB."""
        self.entities.add(entity1)
        self.entities.add(entity2)
        self.relations.add(relation)
        self.entity_to_paths[entity1].append(Path(relation, entity2))
    
    def getPathsFrom(self, entity: str) -> List[Path]:
        """Get all paths from a given entity."""
        return self.entity_to_paths[entity]

def bfs_two_way(e1: str, e2: str, path: List[str], kb: KB, kb_inv: KB) -> bool:
    """Bidirectional search for reasoning, equivalent to the original TensorFlow implementation."""
    start = 0
    end = len(path)
    left = set()
    right = set()
    left.add(e1)
    right.add(e2)
    
    while start < end:
        left_step = path[start]
        left_next = set()
        right_step = path[end-1]
        right_next = set()
        
        if len(left) < len(right):
            start += 1
            for entity in left:
                try:
                    for path_ in kb.getPathsFrom(entity):
                        if path_.relation == left_step:
                            left_next.add(path_.connected_entity)
                except Exception:
                    return False
            left = left_next
        else:
            end -= 1
            for entity in right:
                try:
                    for path_ in kb_inv.getPathsFrom(entity):
                        if path_.relation == right_step:
                            right_next.add(path_.connected_entity)
                except Exception:
                    return False
            right = right_next
    
    if len(right & left) != 0:
        return True
    return False

def teacher(e1: str, e2: str, num_paths: int, env, graphpath: str) -> List[List[Any]]:
    """
    Find paths between entities using BFS with intermediate nodes.
    This function serves as a teacher for reinforcement learning.
    
    Args:
        e1: Start entity
        e2: Target entity
        num_paths: Number of paths to find
        env: Environment
        graphpath: Path to graph file
    
    Returns:
        List of paths between e1 and e2
    """
    # Create knowledge bases
    kb1 = KB()
    kb2 = KB()
    
    # Load graph
    with open(graphpath, 'r') as f:
        lines = f.readlines()
    
    # Add relations to knowledge bases
    for line in lines:
        parts = line.rstrip().split()
        if len(parts) != 3:
            continue
        
        e1_, rel, e2_ = parts
        kb1.addRelation(e1_, rel, e2_)
        kb2.addRelation(e2_, rel, e1_)
    
    # Try to remove direct paths between e1 and e2 (similar to original implementation)
    for path in kb1.getPathsFrom(e1):
        if path.connected_entity == e2:
            kb1.entity_to_paths[e1] = [p for p in kb1.entity_to_paths[e1] if p.connected_entity != e2]
            
    for path in kb2.getPathsFrom(e2):
        if path.connected_entity == e1:
            kb2.entity_to_paths[e2] = [p for p in kb2.entity_to_paths[e2] if p.connected_entity != e1]
    
    # Try the improved bidirectional search with intermediate nodes
    all_paths = []
    
    # First, try direct BFS (maybe there's still a good indirect path)
    direct_paths = bfs(e1, e2, num_paths, kb1, kb2, env)
    if direct_paths:
        all_paths.extend(direct_paths)
    
    # If we need more paths, try with random intermediates
    if len(all_paths) < num_paths:
        # Get a list of random intermediate entities
        all_entities = list(kb1.entities)
        
        # If we have enough entities, pick some random ones
        if len(all_entities) > 2:
            # Remove e1 and e2 from candidates
            if e1 in all_entities:
                all_entities.remove(e1)
            if e2 in all_entities:
                all_entities.remove(e2)
                
            # Try a few random intermediate entities
            import random
            random.seed(42)  # For reproducibility
            num_to_try = min(5, len(all_entities))
            intermediates = random.sample(all_entities, num_to_try)
            
            for intermediate in intermediates:
                # Find path from e1 to intermediate
                path1 = bfs(e1, intermediate, 1, kb1, kb2, env)
                # Find path from intermediate to e2
                path2 = bfs(intermediate, e2, 1, kb1, kb2, env)
                
                # If both paths exist, combine them
                if path1 and path2:
                    # Combine the two paths (remove duplicate state at intermediate)
                    for p1 in path1:
                        for p2 in path2:
                            # Create a combined path
                            combined_path = p1.copy()
                            # Add all transitions from p2 except the first (to avoid duplication)
                            combined_path.extend(p2)
                            all_paths.append(combined_path)
                
                # Stop if we have enough paths
                if len(all_paths) >= num_paths:
                    break
    
    # Return the paths found
    return all_paths[:num_paths]

def bfs(e1: str, e2: str, num_paths: int, kb1: KB, kb2: KB, env) -> List[List[Any]]:
    """
    Breadth-first search to find paths.
    
    Args:
        e1: Start entity
        e2: Target entity
        num_paths: Maximum number of paths to find
        kb1: Forward knowledge base
        kb2: Reverse knowledge base
        env: Environment for transitions
    
    Returns:
        List of paths between e1 and e2
    """
    # Initialize
    num_found_paths = 0
    paths = []
    
    # Initialize queue with start entity
    q = queue.Queue()
    q.put((e1, []))  # (entity, path_so_far)
    
    # Track visited entities with their paths
    visited = defaultdict(list)
    visited[e1] = [[]]
    
    while not q.empty() and num_found_paths < num_paths:
        current, current_path = q.get()
        
        if current == e2:
            # Found a path
            num_found_paths += 1
            paths.append(current_path)
            if num_found_paths >= num_paths:
                break
            continue
        
        # Check path length
        if len(current_path) >= 10:  # Limit to prevent very long paths
            continue
        
        # Get next steps from kb1
        for path in kb1.getPathsFrom(current):
            next_entity = path.connected_entity
            next_relation = path.relation
            
            # Skip if this would create a cycle
            skip = False
            for visit_path in visited[next_entity]:
                if len(visit_path) <= len(current_path):
                    skip = True
                    break
            
            if skip:
                continue
            
            # Create next state
            if env is not None:
                state_idx = [env.entity2id_[current], env.entity2id_[e2], 0]
                for i, r_ in enumerate(env.relations):
                    if r_ == next_relation:
                        action = i
                        break
                else:
                    continue  # Skip if relation not found
                
                state_vec = env.idx_state(state_idx)
                next_state_idx = env.get_next_state(state_idx, action)
                if next_state_idx[0] == 0:  # Invalid transition
                    continue
                
                next_path = current_path + [(state_vec, action, next_state_idx)]
                visited[next_entity].append(next_path)
                q.put((next_entity, next_path))
            else:
                # Simpler path without environment
                next_path = current_path + [(current, next_relation, next_entity)]
                visited[next_entity].append(next_path)
                q.put((next_entity, next_path))
    
    return paths