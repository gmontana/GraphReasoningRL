import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import time
import os
import logging
from collections import defaultdict
import torch.nn.functional as F

def lse(x):
    """Log-sum-exp function for numerical stability"""
    if isinstance(x, list):
        x = torch.tensor(x)
    return torch.logsumexp(x, dim=0).item()

from model.agent import Agent
from model.environment import Environment
from model.baseline import ReactiveBaseline
from utils import get_device


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, params):
        # Transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)
            
        self.device = get_device(params.get('device'))
        params['device'] = self.device
        
        # Initialize agent
        self.agent = Agent(params).to(self.device)
        
        # Initialize environments
        self.train_environment = Environment(params, 'train')
        self.dev_test_environment = Environment(params, 'dev')
        self.test_test_environment = Environment(params, 'test')
        self.test_environment = self.dev_test_environment
        
        # Vocabularies
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        
        # Training parameters
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        
        # Optimizer and baseline
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, 
                                   weight_decay=getattr(self, 'l2_reg_const', 1e-2))
        
        # For decaying beta
        self.global_step = 0
        self.decaying_beta_rate = 0.90
        
        # Initialize pretrained embeddings if provided
        self.initialize_pretrained_embeddings()
        
    def calc_reinforce_loss(self, per_example_loss, per_example_logits, cum_discounted_reward):
        """Calculate REINFORCE loss with baseline and entropy regularization"""
        
        # Stack losses [B, T]
        loss = torch.stack(per_example_loss, dim=1)
        
        # Get baseline value
        baseline_value = self.baseline.get_baseline_value()
        
        # Calculate advantage
        final_reward = cum_discounted_reward - baseline_value
        
        # Normalize rewards (match TensorFlow exactly)
        reward_mean = final_reward.mean(dim=[0, 1], keepdim=True)
        reward_var = final_reward.var(dim=[0, 1], keepdim=True, unbiased=False)
        reward_std = torch.sqrt(reward_var) + 1e-6
        final_reward = (final_reward - reward_mean) / reward_std
        
        # Multiply loss with advantage
        loss = loss * final_reward
        
        # Calculate entropy regularization
        entropy_loss = self.entropy_reg_loss(per_example_logits)
        
        # Total loss
        decaying_beta = self.beta * (self.decaying_beta_rate ** (self.global_step / 200))
        total_loss = loss.mean() - decaying_beta * entropy_loss
        
        return total_loss
    
    def entropy_reg_loss(self, all_logits):
        """Calculate entropy regularization loss"""
        # Stack logits [B, MAX_NUM_ACTIONS, T]
        all_logits = torch.stack(all_logits, dim=2)
        
        # Calculate entropy
        probs = torch.exp(all_logits)
        entropy = -torch.mean(torch.sum(probs * all_logits, dim=1))
        
        return entropy
    
    def calc_cum_discounted_reward(self, rewards):
        """Calculate cumulative discounted reward"""
        batch_size = rewards.shape[0]
        running_add = np.zeros([batch_size])  # [B]
        cum_disc_reward = np.zeros([batch_size, self.path_length])  # [B, T]
        
        # Set last time step to final reward
        cum_disc_reward[:, self.path_length - 1] = rewards
        
        # Calculate cumulative discounted rewards backwards
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
            
        return cum_disc_reward
    
    def initialize_pretrained_embeddings(self):
        """Initialize pretrained embeddings if provided."""
        pretrained_action = getattr(self, 'pretrained_embeddings_action', '')
        pretrained_entity = getattr(self, 'pretrained_embeddings_entity', '')
        
        if pretrained_action and pretrained_action.strip():
            try:
                logger.info(f"Loading pretrained relation embeddings from {pretrained_action}")
                embeddings = np.loadtxt(pretrained_action)
                embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
                
                # Check dimensions match
                if embeddings_tensor.shape != self.agent.relation_embeddings.weight.shape:
                    logger.warning(f"Pretrained relation embedding shape {embeddings_tensor.shape} "
                                 f"doesn't match model shape {self.agent.relation_embeddings.weight.shape}")
                else:
                    self.agent.relation_embeddings.weight.data.copy_(embeddings_tensor)
                    logger.info("Successfully loaded pretrained relation embeddings")
            except Exception as e:
                logger.error(f"Failed to load pretrained relation embeddings: {e}")
        
        if pretrained_entity and pretrained_entity.strip():
            try:
                logger.info(f"Loading pretrained entity embeddings from {pretrained_entity}")
                embeddings = np.loadtxt(pretrained_entity)
                embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
                
                # Check dimensions match
                if embeddings_tensor.shape != self.agent.entity_embeddings.weight.shape:
                    logger.warning(f"Pretrained entity embedding shape {embeddings_tensor.shape} "
                                 f"doesn't match model shape {self.agent.entity_embeddings.weight.shape}")
                else:
                    self.agent.entity_embeddings.weight.data.copy_(embeddings_tensor)
                    logger.info("Successfully loaded pretrained entity embeddings")
            except Exception as e:
                logger.error(f"Failed to load pretrained entity embeddings: {e}")
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.total_iterations
            
        train_loss = 0.0
        self.batch_counter = 0
        
        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1
            
            # Prepare inputs
            query_relation = torch.LongTensor(episode.get_query_relation()).to(self.device)
            state = episode.get_state()
            range_arr = torch.arange(self.batch_size * self.num_rollouts).to(self.device)
            
            # Collect trajectory step by step (like TensorFlow)
            candidate_relation_sequence = []
            candidate_entity_sequence = []
            current_entities_sequence = []
            all_log_probs = []
            all_action_idx = []
            
            # Initialize agent state
            batch_size = self.batch_size * self.num_rollouts
            h = torch.zeros(self.agent.LSTM_Layers, batch_size, 
                           self.agent.m * self.agent.hidden_size).to(self.device)
            c = torch.zeros(self.agent.LSTM_Layers, batch_size,
                           self.agent.m * self.agent.hidden_size).to(self.device)
            lstm_state = (h, c)
            
            # Initialize previous relation
            prev_relation = torch.ones(batch_size, dtype=torch.long).to(self.device) * self.agent.dummy_start_r
            
            # Get query embedding once
            query_embedding = self.agent.relation_embeddings(query_relation)
            
            # Execute trajectory step by step
            for i in range(self.path_length):
                # Current state
                next_relations = torch.LongTensor(state['next_relations']).to(self.device)
                next_entities = torch.LongTensor(state['next_entities']).to(self.device)
                current_entities = torch.LongTensor(state['current_entities']).to(self.device)
                
                # Store for later use
                candidate_relation_sequence.append(next_relations)
                candidate_entity_sequence.append(next_entities)
                current_entities_sequence.append(current_entities)
                
                # One step forward
                lstm_state, log_probs, action_idx, chosen_relation = self.agent.step(
                    next_relations, next_entities, lstm_state, prev_relation,
                    query_embedding, current_entities, range_arr
                )
                
                all_log_probs.append(log_probs)
                all_action_idx.append(action_idx)
                
                # Update environment
                if i < self.path_length - 1:
                    state = episode(action_idx.cpu().numpy())
                
                # Update previous relation
                prev_relation = chosen_relation
            
            # Get rewards
            rewards = episode.get_reward()
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
            cum_discounted_reward = torch.FloatTensor(cum_discounted_reward).to(self.device)
            
            # Calculate losses for each step
            per_example_loss = []
            for i in range(self.path_length):
                log_probs = all_log_probs[i]
                actions = all_action_idx[i]
                
                # Negative log likelihood
                loss_i = -log_probs[range(log_probs.size(0)), actions]
                per_example_loss.append(loss_i)
            
            # Calculate total loss
            loss = self.calc_reinforce_loss(per_example_loss, all_log_probs, cum_discounted_reward)
            
            # Update baseline
            self.baseline.update(cum_discounted_reward.mean().item())
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            
            # Update global step
            self.global_step += 1
            
            # Statistics
            train_loss = 0.98 * train_loss + 0.02 * loss.item()
            avg_reward = np.mean(rewards)
            
            # Calculate hit rate
            reward_reshape = rewards.reshape(self.batch_size, self.num_rollouts)
            reward_reshape = np.sum(reward_reshape, axis=1)
            num_ep_correct = np.sum(reward_reshape > 0)
            
            if self.batch_counter % 100 == 0:
                logger.info(f"Batch {self.batch_counter}: "
                           f"hits={np.sum(rewards):.4f}, "
                           f"avg_reward={avg_reward:.4f}, "
                           f"num_correct={num_ep_correct}, "
                           f"accuracy={num_ep_correct/self.batch_size:.4f}, "
                           f"loss={train_loss:.4f}")
            
            # Evaluation
            if self.batch_counter % self.eval_every == 0:
                self.test(beam=True, save_model=True)
                
            if self.batch_counter >= num_epochs:
                break
                
    def test(self, beam=False, print_paths=False, save_model=True):
        """Test the model"""
        self.agent.eval()
        
        batch_counter = 0
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0
        
        # For NELL evaluation
        paths = defaultdict(list)
        answers = []
        
        total_examples = self.test_environment.total_no_examples
        
        with torch.no_grad():
            for episode in tqdm(self.test_environment.get_episodes()):
                batch_counter += 1
                temp_batch_size = episode.no_examples
                
                query_relation = torch.LongTensor(episode.get_query_relation()).to(self.device)
                
                # Initialize beam search
                if beam:
                    k = self.test_rollouts
                    beam_probs = torch.zeros((temp_batch_size * k, 1)).to(self.device)
                
                # Get initial state
                state = episode.get_state()
                
                # Initialize LSTM state
                h = torch.zeros(self.agent.LSTM_Layers, temp_batch_size * self.test_rollouts, 
                               self.agent.m * self.agent.hidden_size).to(self.device)
                c = torch.zeros(self.agent.LSTM_Layers, temp_batch_size * self.test_rollouts,
                               self.agent.m * self.agent.hidden_size).to(self.device)
                lstm_state = (h, c)
                
                # Initialize previous relation
                prev_relation = torch.ones(temp_batch_size * self.test_rollouts, 
                                          dtype=torch.long).to(self.device) * self.agent.dummy_start_r
                
                # Get query embedding once
                query_embedding = self.agent.relation_embeddings(query_relation)
                
                log_probs = torch.zeros((temp_batch_size * self.test_rollouts,)).to(self.device)
                
                # Path trajectory
                entity_trajectory = []
                relation_trajectory = []
                
                # Walk through path
                for i in range(self.path_length):
                    # Prepare inputs
                    next_relations = torch.LongTensor(state['next_relations']).to(self.device)
                    next_entities = torch.LongTensor(state['next_entities']).to(self.device)
                    current_entities = torch.LongTensor(state['current_entities']).to(self.device)
                    range_arr = torch.arange(current_entities.size(0)).to(self.device)
                    
                    # One step
                    lstm_state, step_log_probs, action_idx, chosen_relation = self.agent.step(
                        next_relations, next_entities, lstm_state, prev_relation,
                        query_embedding, current_entities, range_arr
                    )
                    
                    if beam:
                        # Beam search
                        new_scores = step_log_probs + beam_probs
                        
                        if i == 0:
                            # First step: select top k
                            values, idx = new_scores.topk(k, dim=1)
                            idx = idx.view(-1)
                        else:
                            # Later steps: select top k across all beams
                            new_scores_flat = new_scores.view(temp_batch_size, -1)
                            values, idx = new_scores_flat.topk(k, dim=1)
                            idx = idx.view(-1)
                            
                            # Convert to beam and action indices
                            beam_idx = idx // self.max_num_actions
                            action_idx = idx % self.max_num_actions
                            
                            # Expand beam index for all examples
                            beam_offset = torch.arange(temp_batch_size).to(self.device) * k
                            beam_offset = beam_offset.repeat_interleave(k)
                            beam_idx = beam_idx + beam_offset
                            
                            # Reorder states
                            current_entities = current_entities[beam_idx]
                            next_relations = next_relations[beam_idx]
                            next_entities = next_entities[beam_idx]
                            h, c = lstm_state
                            h = h[:, beam_idx, :]
                            c = c[:, beam_idx, :]
                            lstm_state = (h, c)
                            
                            # Get chosen relations
                            batch_idx = torch.arange(next_relations.size(0)).to(self.device)
                            chosen_relation = next_relations[batch_idx, action_idx]
                            
                            # Update beam probabilities
                            beam_probs = new_scores.view(-1)[idx].unsqueeze(1)
                            
                            if print_paths:
                                # Reorder trajectories
                                for j in range(i):
                                    entity_trajectory[j] = entity_trajectory[j][beam_idx]
                                    relation_trajectory[j] = relation_trajectory[j][beam_idx]
                    
                    # Update for next step
                    prev_relation = chosen_relation
                    
                    if print_paths:
                        entity_trajectory.append(current_entities.cpu().numpy())
                        relation_trajectory.append(chosen_relation.cpu().numpy())
                    
                    # Take action in environment
                    state = episode(action_idx.cpu().numpy())
                    
                    if beam:
                        log_probs = beam_probs.squeeze()
                    else:
                        log_probs += step_log_probs[range(step_log_probs.size(0)), action_idx]
                
                # Get rewards
                rewards = episode.get_reward()
                reward_reshape = rewards.reshape(temp_batch_size, self.test_rollouts)
                log_probs = log_probs.reshape(temp_batch_size, self.test_rollouts).cpu().numpy()
                
                # Calculate metrics
                sorted_idx = np.argsort(-log_probs, axis=1)
                
                # Get current entities for this batch
                ce = episode.state['current_entities'].reshape(temp_batch_size, self.test_rollouts)
                
                for b in range(temp_batch_size):
                    answer_pos = None
                    seen = set()
                    pos = 0
                    
                    if self.pool == 'max':
                        # Max pooling (original logic)
                        for r in sorted_idx[b]:
                            if reward_reshape[b, r] == self.positive_reward:
                                answer_pos = pos
                                break
                            if ce[b, r] not in seen:
                                seen.add(ce[b, r])
                                pos += 1
                    
                    elif self.pool == 'sum':
                        # Sum pooling with log-sum-exp
                        scores = defaultdict(list)
                        answer = ''
                        for r in sorted_idx[b]:
                            scores[ce[b, r]].append(log_probs[b, r])
                            if reward_reshape[b, r] == self.positive_reward:
                                answer = ce[b, r]
                        
                        final_scores = defaultdict(float)
                        for e in scores:
                            final_scores[e] = lse(scores[e])
                        
                        sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                        if answer in sorted_answers:
                            answer_pos = sorted_answers.index(answer)
                        else:
                            answer_pos = None
                    
                    # Update metrics
                    if answer_pos is not None:
                        if answer_pos < 20:
                            all_final_reward_20 += 1
                            if answer_pos < 10:
                                all_final_reward_10 += 1
                                if answer_pos < 5:
                                    all_final_reward_5 += 1
                                    if answer_pos < 3:
                                        all_final_reward_3 += 1
                                        if answer_pos < 1:
                                            all_final_reward_1 += 1
                        
                        auc += 1.0 / (answer_pos + 1)
                    
                    # Collect data for NELL evaluation if print_paths is enabled
                    if print_paths:
                        qr = self.rev_relation_vocab[episode.get_query_relation()[b * self.test_rollouts]]
                        start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                        end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                        
                        paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                        paths[str(qr)].append("Reward:" + str(1 if answer_pos is not None and answer_pos < 10 else 0) + "\n")
                        
                        for r in sorted_idx[b]:
                            indx = b * self.test_rollouts + r
                            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
                            
                            if reward_reshape[b, r] == self.positive_reward:
                                rev = 1
                            else:
                                rev = -1
                            
                            answers.append(self.rev_entity_vocab[se[b, r]] + '\t' + 
                                         self.rev_entity_vocab[ce[b, r]] + '\t' + 
                                         str(log_probs[b, r]) + '\n')
                            
                            if len(entity_trajectory) > 0:
                                entity_path = '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in entity_trajectory])
                                relation_path = '\t'.join([str(self.rev_relation_vocab[re[indx]]) for re in relation_trajectory])
                                paths[str(qr)].append(entity_path + '\n' + relation_path + '\n' + 
                                                    str(rev) + '\n' + str(log_probs[b, r]) + '\n___\n')
                        
                        paths[str(qr)].append("#####################\n")
        
        # Calculate final metrics
        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples
        
        # Save model if improved
        if save_model and all_final_reward_10 >= self.max_hits_at_10:
            self.max_hits_at_10 = all_final_reward_10
            self.save_model()
        
        # Save paths and answers for NELL evaluation
        if print_paths:
            import codecs
            test_beam_dir = os.path.join(self.output_dir, 'test_beam')
            os.makedirs(test_beam_dir, exist_ok=True)
            
            logger.info(f"Printing paths at {test_beam_dir}")
            for q in paths:
                j = q.replace('/', '-')
                path_file = os.path.join(test_beam_dir, f'paths_{j}')
                with codecs.open(path_file, 'w', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            
            answers_file = os.path.join(test_beam_dir, 'pathsanswers')
            with open(answers_file, 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)
        
        # Log results
        logger.info(f"Hits@1: {all_final_reward_1:.4f}")
        logger.info(f"Hits@3: {all_final_reward_3:.4f}")
        logger.info(f"Hits@5: {all_final_reward_5:.4f}")
        logger.info(f"Hits@10: {all_final_reward_10:.4f}")
        logger.info(f"Hits@20: {all_final_reward_20:.4f}")
        logger.info(f"MRR: {auc:.4f}")
        
        self.agent.train()
        
        return {
            'hits@1': all_final_reward_1,
            'hits@3': all_final_reward_3,
            'hits@5': all_final_reward_5,
            'hits@10': all_final_reward_10,
            'hits@20': all_final_reward_20,
            'mrr': auc
        }
    
    def save_model(self):
        """Save model checkpoint"""
        save_path = os.path.join(self.model_dir, 'model.pt')
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'max_hits_at_10': self.max_hits_at_10
        }, save_path)
        logger.info(f"Model saved to {save_path}")
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.max_hits_at_10 = checkpoint['max_hits_at_10']
        logger.info(f"Model loaded from {path}")