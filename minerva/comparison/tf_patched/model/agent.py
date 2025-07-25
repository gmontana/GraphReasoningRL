import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()
from model.lstm_wrapper import LSTMWrapper


class Agent(object):

    def __init__(self, params):

        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf_v1.constant(params['entity_vocab']['PAD'], dtype=tf_v1.int32)
        self.rPAD = tf_v1.constant(params['relation_vocab']['PAD'], dtype=tf_v1.int32)
        if params['use_entity_embeddings']:
            self.entity_initializer = tf_v1.initializers.glorot_uniform()
        else:
            self.entity_initializer = tf_v1.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf_v1.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        with tf_v1.variable_scope("action_lookup_table"):
            self.action_embedding_placeholder = tf_v1.placeholder(tf_v1.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            self.relation_lookup_table = tf_v1.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf_v1.float32,
                                                         initializer=tf_v1.initializers.glorot_uniform(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf_v1.variable_scope("entity_lookup_table"):
            self.entity_embedding_placeholder = tf_v1.placeholder(tf_v1.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf_v1.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf_v1.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        with tf_v1.variable_scope("policy_step"):
            # Use custom LSTM wrapper for compatibility
            self.policy_step = LSTMWrapper(
                hidden_size=self.m * self.hidden_size,
                num_layers=self.LSTM_Layers
            )

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def policy_MLP(self, state):
        with tf_v1.variable_scope("MLP_for_policy"):
            hidden = tf_v1.keras.layers.Dense(4 * self.hidden_size, activation='relu')(state)
            output = tf_v1.keras.layers.Dense(self.m * self.embedding_size, activation='relu')(hidden)
        return output

    def action_encoder(self, next_relations, next_entities):
        with tf_v1.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf_v1.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf_v1.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf_v1.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
             label_action, range_arr, first_step_of_test):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        # 1. one step of rnn
        output, new_state = self.policy_step(prev_action_embedding, prev_state)  # output: [B, 4D]

        # Get state vector
        prev_entity = tf_v1.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        if self.use_entity_embeddings:
            state = tf_v1.concat([output, prev_entity], axis=-1)
        else:
            state = output
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        state_query_concat = tf_v1.concat([state, query_embedding], axis=-1)

        # MLP for policy#

        output = self.policy_MLP(state_query_concat)
        output_expanded = tf_v1.expand_dims(output, axis=1)  # [B, 1, 2D]
        prelim_scores = tf_v1.reduce_sum(tf_v1.multiply(candidate_action_embeddings, output_expanded), axis=2)

        # Masking PAD actions

        comparison_tensor = tf_v1.ones_like(next_relations, dtype=tf_v1.int32) * self.rPAD  # matrix to compare
        mask = tf_v1.equal(next_relations, comparison_tensor)  # The mask
        dummy_scores = tf_v1.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = tf_v1.where(mask, dummy_scores, prelim_scores)  # [B, MAX_NUM_ACTIONS]

        # 4 sample action
        action = tf_v1.to_int32(tf_v1.multinomial(logits=scores, num_samples=1))  # [B, 1]

        # loss
        # 5a.
        label_action =  tf_v1.squeeze(action, axis=1)
        loss = tf_v1.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]

        # 6. Map back to true id
        action_idx = tf_v1.squeeze(action)
        chosen_relation = tf_v1.gather_nd(next_relations, tf_v1.transpose(tf_v1.stack([range_arr, action_idx])))

        return loss, new_state, tf_v1.nn.log_softmax(scores), action_idx, chosen_relation

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities,
                 path_label, query_relation, range_arr, first_step_of_test, T=3, entity_sequence=0):

        self.baseline_inputs = []
        # get the query vector
        query_embedding = tf_v1.nn.embedding_lookup(self.relation_lookup_table, query_relation)  # [B, 2D]
        state = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf_v1.float32)

        prev_relation = self.dummy_start_label

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []

        with tf_v1.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_possible_relations = candidate_relation_sequence[t]  # [B, MAX_NUM_ACTIONS, MAX_EDGE_LENGTH]
                next_possible_entities = candidate_entity_sequence[t]
                current_entities_t = current_entities[t]

                path_label_t = path_label[t]  # [B]

                loss, state, logits, idx, chosen_relation = self.step(next_possible_relations,
                                                                              next_possible_entities,
                                                                              state, prev_relation, query_embedding,
                                                                              current_entities_t,
                                                                              label_action=path_label_t,
                                                                              range_arr=range_arr,
                                                                              first_step_of_test=first_step_of_test)

                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(idx)
                prev_relation = chosen_relation

            # [(B, T), 4D]

        return all_loss, all_logits, action_idx
