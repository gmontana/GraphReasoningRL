import numpy as np
import torch
from data.feed_data import RelationEntityBatcher
from data.grapher import RelationEntityGrapher


class Episode:
    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, \
            self.positive_reward, self.negative_reward, self.mode, self.batcher = params
        if self.mode == 'train':
            self.num_rollouts = self.num_rollouts
        else:
            self.num_rollouts = self.test_rollouts
            
        self.current_hop = 0
        start_entities, query_relation, end_entities, all_answers = data
        self.no_examples = start_entities.shape[0]
        
        # Repeat for rollouts
        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers
        
        # Get initial actions
        next_actions = self.grapher.return_next_actions(
            self.current_entities, self.start_entities, self.query_relation,
            self.end_entities, self.all_answers, 
            self.current_hop == self.path_len - 1,
            self.num_rollouts
        )
        
        self.state = {
            'next_relations': next_actions[:, :, 1],
            'next_entities': next_actions[:, :, 0],
            'current_entities': self.current_entities
        }
        
    def get_state(self):
        return self.state
    
    def get_query_relation(self):
        return self.query_relation
    
    def get_reward(self):
        """Calculate rewards for current entities"""
        reward = (self.current_entities == self.end_entities)
        
        # Convert boolean to reward values
        reward = np.where(reward, self.positive_reward, self.negative_reward)
        return reward
    
    def __call__(self, action):
        """Take action and return new state"""
        self.current_hop += 1
        
        # Update current entities based on action
        batch_idx = np.arange(self.no_examples * self.num_rollouts)
        self.current_entities = self.state['next_entities'][batch_idx, action]
        
        # Get next possible actions
        next_actions = self.grapher.return_next_actions(
            self.current_entities, self.start_entities, self.query_relation,
            self.end_entities, self.all_answers,
            self.current_hop == self.path_len - 1,
            self.num_rollouts
        )
        
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        
        return self.state


class Environment:
    def __init__(self, params, mode='train'):
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        
        input_dir = params['data_input_dir']
        
        # Initialize batcher
        if mode == 'train':
            self.batcher = RelationEntityBatcher(
                input_dir=input_dir,
                batch_size=params['batch_size'],
                entity_vocab=params['entity_vocab'],
                relation_vocab=params['relation_vocab']
            )
        else:
            self.batcher = RelationEntityBatcher(
                input_dir=input_dir,
                mode=mode,
                batch_size=params['batch_size'],
                entity_vocab=params['entity_vocab'],
                relation_vocab=params['relation_vocab']
            )
            self.total_no_examples = self.batcher.store.shape[0]
            
        # Initialize grapher
        self.grapher = RelationEntityGrapher(
            triple_store=params['data_input_dir'] + '/' + 'graph.txt',
            max_num_actions=params['max_num_actions'],
            entity_vocab=params['entity_vocab'],
            relation_vocab=params['relation_vocab']
        )
        
    def get_episodes(self):
        """Yield episodes for training/testing"""
        params = (self.batch_size, self.path_len, self.num_rollouts, 
                  self.test_rollouts, self.positive_reward, self.negative_reward,
                  self.mode, self.batcher)
        
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data is None:
                    return
                yield Episode(self.grapher, data, params)