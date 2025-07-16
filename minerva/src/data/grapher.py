import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RelationEntityGrapher:
    def __init__(self, triple_store, max_num_actions, entity_vocab, relation_vocab):
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.max_num_actions = max_num_actions
        
        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.ePAD_id = self.entity_vocab['PAD']
        self.rPAD_id = self.relation_vocab['PAD']
        
        # Create reverse vocabularies
        self.rev_entity_vocab = {v: k for k, v in entity_vocab.items()}
        self.rev_relation_vocab = {v: k for k, v in relation_vocab.items()}
        
        # Build knowledge graph
        self.store = self.load_graph_from_triple_store(triple_store)
        
    def load_graph_from_triple_store(self, triple_store):
        """Load knowledge graph from file"""
        logger.info(f"Loading knowledge graph from {triple_store}")
        
        # Dictionary to store graph: entity -> list of (entity, relation) tuples
        store = defaultdict(list)
        
        with open(triple_store, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    e1, r, e2 = line.split('\t')
                    
                    # Convert to IDs
                    e1_id = self.entity_vocab.get(e1, self.ePAD_id)
                    r_id = self.relation_vocab.get(r, self.rPAD_id)
                    e2_id = self.entity_vocab.get(e2, self.ePAD_id)
                    
                    # Add edge to graph
                    store[e1_id].append((e2_id, r_id))
                    
        # Convert to regular dict
        store = dict(store)
        
        logger.info(f"Loaded graph with {len(store)} entities")
        return store
    
    def return_next_actions(self, current_entities, start_entities, query_relations,
                           end_entities, all_answers, last_step, num_rollouts):
        """
        Get possible next actions from current entities
        Returns: [batch_size, max_num_actions, 2] where [:,:,0] are entities and [:,:,1] are relations
        """
        batch_size = current_entities.shape[0]
        
        # Placeholder for next actions
        next_actions = np.ones((batch_size, self.max_num_actions, 2), dtype=np.int64)
        next_actions[:, :, 0] = self.ePAD_id  # entity padding
        next_actions[:, :, 1] = self.rPAD_id  # relation padding
        
        for i in range(batch_size):
            current_e = current_entities[i]
            
            if current_e in self.store:
                # Get all possible actions from this entity
                actions = self.store[current_e]
                
                # Limit to max_num_actions
                if len(actions) > self.max_num_actions:
                    # Randomly sample if too many actions
                    indices = np.random.choice(len(actions), self.max_num_actions, replace=False)
                    actions = [actions[idx] for idx in indices]
                
                # Fill in the actions
                for j, (next_e, next_r) in enumerate(actions):
                    if j >= self.max_num_actions:
                        break
                    next_actions[i, j, 0] = next_e
                    next_actions[i, j, 1] = next_r
                    
        return next_actions