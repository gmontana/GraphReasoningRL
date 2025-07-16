#!/usr/bin/python

import sys
import numpy as np
from BFS.KB import *
from sklearn import linear_model
# Use tensorflow.keras instead of standalone keras
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation

relation = sys.argv[1]

# Fix the path to use absolute references from the repository root
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dataPath_ = os.path.join(repo_root, 'NELL-995/tasks', relation)
featurePath = os.path.join(dataPath_, 'path_to_use.txt')
feature_stats = os.path.join(dataPath_, 'path_stats.txt')
relationId_path = os.path.join(repo_root, 'NELL-995/relation2id.txt')

def train(kb, kb_inv, named_paths):
	try:
		f = open(os.path.join(dataPath_, 'train.pairs'))
		train_data = f.readlines()
		f.close()
	except Exception as e:
		print(f"Could not find train.pairs, using train_pos instead: {e}")
		f = open(os.path.join(dataPath_, 'train_pos'))
		train_data = f.readlines()
		f.close()
	
	train_pairs = []
	train_labels = []
	for line in train_data:
		line = line.strip()
		if not line or ',' not in line:
			continue
		
		parts = line.split(',')
		if len(parts) < 2:
			continue
			
		e1 = parts[0].replace('thing$','')
		
		# Handle different formats
		if ':' in parts[1]:
			e2 = parts[1].split(':')[0].replace('thing$','')
		else:
			e2 = parts[1].replace('thing$','')
			
		if (e1 not in kb.entities) or (e2 not in kb.entities):
			continue
			
		train_pairs.append((e1,e2))
		label = 1 if (line.endswith('+') or 'concept:athletePlaysForTeam+' in line) else 0
		train_labels.append(label)
	
	training_features = []
	for sample in train_pairs:
		feature = []
		for path in named_paths:
			feature.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
		training_features.append(feature)
	
	model = Sequential()
	input_dim = len(named_paths)
	model.add(Dense(1, activation='sigmoid', input_dim=input_dim))
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
	model.fit(np.array(training_features), np.array(train_labels), epochs=300, batch_size=128)
	return model

def get_features():
	stats = {}
	try:
		f = open(feature_stats)
		path_freq = f.readlines()
		f.close()
		for line in path_freq:
			if '\t' in line:
				path = line.split('\t')[0]
				num = int(line.split('\t')[1])
				stats[path] = num
		if stats:
			max_freq = np.max(list(stats.values()))
		else:
			print("Warning: No paths found in feature_stats file")
			max_freq = 0
	except Exception as e:
		print(f"Warning: Could not read feature_stats: {e}")
		max_freq = 0

	relation2id = {}
	f = open(relationId_path)
	content = f.readlines()
	f.close()
	for line in content:
		relation2id[line.split()[0]] = int(line.split()[1])

	useful_paths = []
	named_paths = []
	f = open(featurePath)
	paths = f.readlines()
	f.close()

	print(len(paths))

	for line in paths:
		path = line.rstrip()

		length = len(path.split(' -> '))

		if length <= 10:
			pathIndex = []
			pathName = []
			relations = path.split(' -> ')

			for rel in relations:
				pathName.append(rel)
				rel_id = relation2id.get(rel, 0)  # Default to 0 if not found
				pathIndex.append(rel_id)
			useful_paths.append(pathIndex)
			named_paths.append(pathName)

	print('How many paths used: ', len(useful_paths))
	return useful_paths, named_paths

def evaluate_logic():
	kb = KB()
	kb_inv = KB()

	f = open(os.path.join(dataPath_, 'graph.txt'))
	kb_lines = f.readlines()
	f.close()

	for line in kb_lines:
		e1 = line.split()[0]
		rel = line.split()[1]
		e2 = line.split()[2]
		kb.addRelation(e1,rel,e2)
		kb_inv.addRelation(e2,rel,e1)

	_, named_paths = get_features()

	model = train(kb, kb_inv, named_paths)

	try:
		f = open(os.path.join(dataPath_, 'sort_test.pairs'))
		test_data = f.readlines()
		f.close()
	except Exception as e:
		print(f"Could not find sort_test.pairs, using train_pos for testing: {e}")
		f = open(os.path.join(dataPath_, 'train_pos'))
		test_data = f.readlines()
		f.close()
	
	test_pairs = []
	test_labels = []
	for line in test_data:
		line = line.strip()
		if not line or ',' not in line:
			continue
		
		parts = line.split(',')
		if len(parts) < 2:
			continue
			
		e1 = parts[0].replace('thing$','')
		
		# Handle different formats
		if ':' in parts[1]:
			e2 = parts[1].split(':')[0].replace('thing$','')
		else:
			e2 = parts[1].replace('thing$','')
			
		if (e1 not in kb.entities) or (e2 not in kb.entities):
			continue
			
		test_pairs.append((e1,e2))
		label = 1 if (line.endswith('+') or 'concept:athletePlaysForTeam+' in line) else 0
		test_labels.append(label)

	aps = []
	query = test_pairs[0][0] if test_pairs else None
	y_true = []
	y_score = []

	score_all = []

	for idx, sample in enumerate(test_pairs):
		if sample[0] == query:
			features = []
			for path in named_paths:
				features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

			score = model.predict(np.reshape(features, [1,-1]))

			score_all.append(score[0][0])
			y_score.append(score[0][0])
			y_true.append(test_labels[idx])
		else:
			query = sample[0]
			count = list(zip(y_score, y_true))
			count.sort(key = lambda x:x[0], reverse=True)
			ranks = []
			correct = 0
			for idx_, item in enumerate(count):
				if item[1] == 1:
					correct +=  1
					ranks.append(correct/(1.0+idx_))
			if len(ranks) == 0:
				aps.append(0)
			else:
				aps.append(np.mean(ranks))
			y_true = []
			y_score = []
			features = []
			for path in named_paths:
				features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

			score = model.predict(np.reshape(features,[1,-1]))

			score_all.append(score[0][0])
			y_score.append(score[0][0])
			y_true.append(test_labels[idx])

	if y_true and y_score:
		count = list(zip(y_score, y_true))
		count.sort(key = lambda x:x[0], reverse=True)
		ranks = []
		correct = 0
		for idx_, item in enumerate(count):
			if item[1] == 1:
				correct +=  1
				ranks.append(correct/(1.0+idx_))
		if ranks:
			aps.append(np.mean(ranks))

	score_label = list(zip(score_all, test_labels))
	score_label_ranked = sorted(score_label, key = lambda x:x[0], reverse=True)

	if aps:
		mean_ap = np.mean(aps)
		print('RL MAP: ', mean_ap)


def bfs_two(e1,e2,path,kb,kb_inv):
	'''the bidirectional search for reasoning'''
	start = 0
	end = len(path)
	left = set()
	right = set()
	left.add(e1)
	right.add(e2)

	left_path = []
	right_path = []
	while(start < end):
		left_step = path[start]
		left_next = set()
		right_step = path[end-1]
		right_next = set()

		if len(left) < len(right):
			left_path.append(left_step)
			start += 1
			for entity in left:
				try:
					for path_ in kb.getPathsFrom(entity):
						if path_.relation == left_step:
							left_next.add(path_.connected_entity)
				except Exception as e:
					# Handle exceptions gracefully
					return False
			left = left_next

		else: 
			right_path.append(right_step)
			end -= 1
			for entity in right:
				try:
					for path_ in kb_inv.getPathsFrom(entity):
						if path_.relation == right_step:
							right_next.add(path_.connected_entity)
				except Exception as e:
					# Handle exceptions gracefully
					return False
			right = right_next

	if len(right & left) != 0:
		return True 
	return False


if __name__ == '__main__':
	evaluate_logic()