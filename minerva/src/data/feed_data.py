import numpy as np
import logging

logger = logging.getLogger(__name__)


class RelationEntityBatcher:
    def __init__(self, input_dir, batch_size, entity_vocab, relation_vocab, mode='train'):
        self.batch_size = batch_size
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.mode = mode
        self.input_dir = input_dir
        
        # Load data
        if mode == 'train':
            self.store = self.load_data(input_dir + '/train.txt')
        elif mode == 'dev':
            self.store = self.load_data(input_dir + '/dev.txt')
        elif mode == 'test':
            self.store = self.load_data(input_dir + '/test.txt')
        else:
            raise ValueError(f"Invalid mode: {mode}")
            
        # Create answer lookup for evaluation
        self.create_answer_lookup()
        
        logger.info(f"Loaded {len(self.store)} {mode} examples")
        
    def load_data(self, filename):
        """Load triples from file"""
        data = []
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    e1, r, e2 = line.split('\t')
                    
                    # Convert to IDs
                    e1_id = self.entity_vocab.get(e1, self.entity_vocab['PAD'])
                    r_id = self.relation_vocab.get(r, self.relation_vocab['PAD'])
                    e2_id = self.entity_vocab.get(e2, self.entity_vocab['PAD'])
                    
                    data.append([e1_id, r_id, e2_id])
                    
        return np.array(data, dtype=np.int64)
    
    def create_answer_lookup(self):
        """Create lookup for all correct answers for each (e1, r) pair"""
        self.all_answers = {}
        
        for e1, r, e2 in self.store:
            key = (e1, r)
            if key not in self.all_answers:
                self.all_answers[key] = set()
            self.all_answers[key].add(e2)
            
    def yield_next_batch_train(self):
        """Yield training batches"""
        while True:
            # Shuffle data
            indices = np.random.permutation(len(self.store))
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                if len(batch_indices) < self.batch_size:
                    # Skip incomplete batches
                    continue
                    
                batch = self.store[batch_indices]
                
                start_entities = batch[:, 0]
                query_relations = batch[:, 1]
                end_entities = batch[:, 2]
                
                # Get all answers for each query
                all_answers = []
                for j in range(len(batch)):
                    key = (start_entities[j], query_relations[j])
                    answers = self.all_answers.get(key, set())
                    all_answers.append(answers)
                
                yield start_entities, query_relations, end_entities, all_answers
                
    def yield_next_batch_test(self):
        """Yield test/dev batches"""
        for i in range(0, len(self.store), self.batch_size):
            batch = self.store[i:i + self.batch_size]
            
            if len(batch) == 0:
                yield None
                continue
                
            start_entities = batch[:, 0]
            query_relations = batch[:, 1]
            end_entities = batch[:, 2]
            
            # Get all answers for each query
            all_answers = []
            for j in range(len(batch)):
                key = (start_entities[j], query_relations[j])
                answers = self.all_answers.get(key, set())
                all_answers.append(answers)
            
            yield start_entities, query_relations, end_entities, all_answers