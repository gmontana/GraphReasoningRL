import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Agent(nn.Module):
    def __init__(self, params):
        super(Agent, self).__init__()
        
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        
        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2
            
        self.device = params.get('device', 'cpu')
        
        # Embeddings
        self.relation_embeddings = nn.Embedding(
            self.action_vocab_size, 
            2 * self.embedding_size
        )
        if params.get('pretrained_embeddings_action') is not None:
            self.relation_embeddings.weight.data = torch.FloatTensor(params['pretrained_embeddings_action'])
        else:
            nn.init.xavier_uniform_(self.relation_embeddings.weight)
        self.relation_embeddings.weight.requires_grad = self.train_relations
        
        if self.use_entity_embeddings:
            self.entity_embeddings = nn.Embedding(
                self.entity_vocab_size,
                2 * self.entity_embedding_size
            )
            if params.get('pretrained_embeddings_entity') is not None:
                self.entity_embeddings.weight.data = torch.FloatTensor(params['pretrained_embeddings_entity'])
            else:
                nn.init.xavier_uniform_(self.entity_embeddings.weight)
            self.entity_embeddings.weight.requires_grad = self.train_entities
        else:
            self.entity_embeddings = nn.Embedding(
                self.entity_vocab_size,
                2 * self.entity_embedding_size
            )
            self.entity_embeddings.weight.data.zero_()
            self.entity_embeddings.weight.requires_grad = False
            
        # LSTM
        self.policy_step = nn.LSTM(
            self.m * self.embedding_size,
            self.m * self.hidden_size,
            num_layers=self.LSTM_Layers,
            batch_first=True
        )
        
        # MLP for policy
        self.policy_mlp = nn.Sequential(
            nn.Linear(self.m * self.hidden_size + 2 * self.embedding_size, 4 * self.hidden_size),
            nn.ReLU(),
            nn.Linear(4 * self.hidden_size, self.m * self.embedding_size),
            nn.ReLU()
        )
        
        # Constants
        self.ePAD = params['entity_vocab']['PAD']
        self.rPAD = params['relation_vocab']['PAD']
        self.dummy_start_r = params['relation_vocab']['DUMMY_START_RELATION']
        
    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)
    
    def action_encoder(self, next_relations, next_entities):
        """Encode actions as relation-entity embeddings"""
        relation_embedding = self.relation_embeddings(next_relations)
        entity_embedding = self.entity_embeddings(next_entities)
        
        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding
            
        return action_embedding
    
    def step(self, next_relations, next_entities, prev_state, prev_relation, 
             query_embedding, current_entities, range_arr):
        """One step of LSTM policy"""
        
        # Get previous action embedding
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        
        # LSTM step
        prev_action_embedding = prev_action_embedding.unsqueeze(1)  # Add sequence dimension
        output, new_state = self.policy_step(prev_action_embedding, prev_state)
        output = output.squeeze(1)  # Remove sequence dimension
        
        # Get state vector
        if self.use_entity_embeddings:
            prev_entity = self.entity_embeddings(current_entities)
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output
            
        # Encode candidate actions
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        
        # Concatenate state and query
        state_query_concat = torch.cat([state, query_embedding], dim=-1)
        
        # MLP for policy
        output = self.policy_mlp(state_query_concat)
        output_expanded = output.unsqueeze(1)  # [B, 1, m*D]
        
        # Score actions
        prelim_scores = torch.sum(candidate_action_embeddings * output_expanded, dim=2)
        
        # Mask PAD actions
        mask = next_relations == self.rPAD
        scores = prelim_scores.masked_fill(mask, -99999.0)
        
        # Sample action
        probs = F.softmax(scores, dim=1)
        action = torch.multinomial(probs, num_samples=1)
        action_idx = action.squeeze(1)
        
        # Get chosen relation
        batch_idx = torch.arange(next_relations.size(0), device=next_relations.device)
        chosen_relation = next_relations[batch_idx, action_idx]
        
        # Log probabilities for loss computation
        log_probs = F.log_softmax(scores, dim=1)
        
        return new_state, log_probs, action_idx, chosen_relation
    
    def forward(self, candidate_relation_sequence, candidate_entity_sequence,
                current_entities, query_relation, range_arr, T=3):
        """
        Forward pass through T steps
        Returns: log_probs, action_indices
        """
        batch_size = candidate_relation_sequence[0].size(0)
        
        # Get query embedding
        query_embedding = self.relation_embeddings(query_relation)
        
        # Initialize LSTM state
        h0 = torch.zeros(self.LSTM_Layers, batch_size, self.m * self.hidden_size).to(self.device)
        c0 = torch.zeros(self.LSTM_Layers, batch_size, self.m * self.hidden_size).to(self.device)
        state = (h0, c0)
        
        # Initialize previous relation
        prev_relation = torch.ones(batch_size, dtype=torch.long).to(self.device) * self.dummy_start_r
        
        all_log_probs = []
        all_action_idx = []
        
        for t in range(T):
            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            current_entities_t = current_entities[t]
            
            state, log_probs, idx, chosen_relation = self.step(
                next_possible_relations,
                next_possible_entities,
                state,
                prev_relation,
                query_embedding,
                current_entities_t,
                range_arr
            )
            
            all_log_probs.append(log_probs)
            all_action_idx.append(idx)
            prev_relation = chosen_relation
            
        return all_log_probs, all_action_idx