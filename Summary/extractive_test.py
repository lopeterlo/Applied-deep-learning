import sys
import os
import json
import spacy
import en_core_web_sm
import numpy as np
import random
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from nltk.tokenize import RegexpTokenizer

import time

batch_size = 64



class SummaryDataset(Dataset):
	def __init__(self, data, tokenizer, w2v, test = False, input_size =300):
		self.data = data
		self.tokenizer = tokenizer
		self.w2v = w2v
		self.test = test
		self.input_size = input_size
		
	def __len__(self):
		return len(self.data)
	
	
	def __getitem__(self, idx):
		if not self.test:
			embeddings, labels = self.tokenize(self.data.iloc[idx])
			return embeddings, labels
			
		else:
			embeddings, id, words_pos, sentence = self.tokenize_test(self.data.iloc[idx])
			return embeddings, id, words_pos, sentence
	
	def tokenize(self, inputs):
		data = inputs['text']
		sent_b = inputs['sent_bounds']
		ans_idx = inputs['extractive_summary']

		all_embeddings = []
		labels = []
		if len(inputs['sent_bounds'])  > 1:
			n_sam_idx = random.randint(0, len(inputs['sent_bounds'])-1)
			while n_sam_idx == ans_idx:
				n_sam_idx = random.randint(0, len(inputs['sent_bounds'])-1)
			
			if n_sam_idx > ans_idx:
				sent_b = [sent_b[ans_idx], sent_b[n_sam_idx]]
				ans_idx = 0
			else:
				sent_b = [sent_b[n_sam_idx], sent_b[ans_idx]]
				ans_idx = 1
		for idx, pos in enumerate(sent_b):
			tokens = self.tokenizer.tokenize(data[pos[0]: pos[1]])
			embeddings, words = self.clean(tokens)
			all_embeddings += embeddings
			if idx == ans_idx:
				labels += [1 for i in range(len(words))]
			else:
				labels += [0 for i in range(len(words))]
		return torch.tensor(all_embeddings), torch.tensor(labels)
	
	def tokenize_test(self, inputs):
		data = inputs['text']
		sent_b = inputs['sent_bounds']
		all_embeddings = []
		id = str(inputs['id'])
		word_pos = []
		all_sentences = []
		for idx, pos in enumerate(sent_b):
			all_sentences.append(data[pos[0]: pos[1]][:-2])
			tokens = self.tokenizer.tokenize(data[pos[0]: pos[1]])
			embeddings, words = self.clean(tokens)
			all_embeddings += embeddings
			word_pos += [idx for i in range(len(words))]
		return torch.tensor(all_embeddings), id, word_pos, all_sentences
	
	def clean(self, tokens):
		embeddings = []
		words = []
		for idx, token in enumerate(tokens):
			token = token.lower()
			if token in self.w2v.keys():
				embeddings.append(self.w2v[token])
				words.append(token)
		if len(embeddings) ==0:
			return [[0.0 for i in range(self.input_size)]], [[]]
		return embeddings, words
  
 

class LSTM_model(nn.Module):
	def __init__(self, input_size, hidden_size = 200, n_layers = 1, bidirectional = False):
		super(LSTM_model, self).__init__()
		self.bidirectional = bidirectional
		self.model = nn.LSTM(input_size, hidden_size, n_layers,  batch_first = True, bidirectional= bidirectional)
		self.relu = nn.ReLU()
		if bidirectional:
			self.l1 = nn.Linear(hidden_size * 2, 64)
			self.l2 = nn.Linear(64, 2)
			# self.l3 = nn.Linear(48, 2)
		else:
			self.l1 = nn.Linear(hidden_size, 64)
			self.l2 = nn.Linear(64, 2)
			# self.l3 = nn.Linear(48, 2)
		self.init_hidden()
		
	def forward(self, x, h= None):
		self.model.flatten_parameters()
		x, (hn, cn) = self.model(x, h)
		l1 = self.l1(x)
		l2 = self.l2(l1)
		# l3 = self.l3(l2)
		# out = Func.softmax(l2, dim=2) # along rows
		return l2

	def init_hidden(self):
		for name, p in self.model.named_parameters():
			if 'weight' in name:
				nn.init.orthogonal_(p)
			elif 'bias' in name:
				nn.init.constant_(p, 0)

class Extractive_model(object):
	def __init__(self, input_size = 300, batch_size = 64, hidden_size = 200 , epoch = 3, bidirectional = True, lr= 1e-3):
		super(Extractive_model, self).__init__()
		self.epoch = epoch
		self.batch_size = batch_size
		self.lr = lr
		self.model = None
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.bidirectional = bidirectional
		self.gpu = torch.cuda.is_available()
	def fit(self, train_data, valid_data):
		self.model = LSTM_model(input_size= self.input_size, hidden_size = self.hidden_size, bidirectional = self.bidirectional)
		if self.gpu:
			self.model = self.model.cuda()
		index = 0
		min_loss = 10000
		best_model = None
		for i in range(self.epoch):
			self.model.train()
			total_loss = 0
			total = 0
			for x, y in train_data:
				if len(x) == 0:
					continue
				if self.gpu:
					x = x.cuda()   # x, y torch.Size([2, 56, 300]) torch.Size([2, 56])
					y = y.cuda()
				pred = self.model(x)  #torch.Size([2,56, 2])
				pred = pred.view(-1, pred.size(2)) #torch.Size([112, 2])
				# w_1 = sum([sum(y[i]) for i in range(len(y))]).tolist()
				# w_0 = sum([len(y[i]) for i in range(len(y))])
				# c0 = w_1/w_0
				# c1 = w_0/w_1
				# position_w = torch.FloatTensor([c0, c1]).cuda()
				# loss_f = nn.BCEWithLogitsLoss(pos_weight = position_w)
				# loss_f = nn.BCEWithLogitsLoss() # [batch, max_len, class]
				y = y.view(-1)
				loss_f = nn.CrossEntropyLoss() # [batch, max_len]
				
				# y = torch.tensor([[0,1] if j == 1 else[1,0] for j in y]).float()
				if self.gpu:
					y = y.cuda()

				loss = loss_f(pred, y)
				loss.backward()
				opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
				opt.step()
				opt.zero_grad()
				index += 1
				total += 1
				total_loss += loss.item()
				print(f'Epoch : {i}, Iteration : {index+1} , loss: {loss.item()}, avg_loss: {total_loss/total} ', end = '\r')
			print('\n')
			valid_loss = self.valid(valid_data)
			if valid_loss < min_loss:
				best_model = self.model
				min_loss = valid_loss
			index = 0
		self.model = best_model
		
	def valid(self, valid_data):
		self.model.eval()
		total, total_loss = 0,0
		index = 0
		print('\n')
		for x, y in valid_data:
			if self.gpu:
				x = x.cuda()   # x, y torch.Size([2, 56, 300]) torch.Size([2, 56])
				y = y.cuda()
			pred = self.model(x)  #torch.Size([2,56, 2])
			pred = pred.view(-1, pred.size(2)) #torch.Size([112, 2])
			y = y.view(-1)
			# y = torch.tensor([[0,1] if j == 1 else[1,0] for j in y]).float()
			if self.gpu:
				y = y.cuda()
			# loss_f = nn.BCEWithLogitsLoss()
			loss_f = nn.CrossEntropyLoss()
			loss = loss_f(pred, y)
			total += len(y)
			total_loss += loss.item() * len(y)
			print(f' validation iteration index {index}, Loss : {loss.item()}', end = '\r')
			index += 1
		print(f'\n validation loss: {total_loss / total} \n', end = '\r')
		print('\n ==========================')

		return total_loss / total


	def test(self, test_data):
		prediction = ''
		for x, id, words_pos, sentence in test_data:

			x = x.cuda()
			pred = self.model(x) # torch.Size([16, 282, 2])
			out = Func.softmax(pred, dim=2) # ([2, 255, 2])
			values, indexs = out.max(-1)
			for idx, pred in enumerate(indexs):
				ans = pred.tolist()
				cum_num = defaultdict(lambda:0,{})
				for (i,j) in zip(words_pos[idx], ans):
					if j == 1:
						cum_num[i] += 1
				cum_num = sorted(cum_num.items(), key= lambda x: x[1], reverse = True)
				extractive_pred = [i for i, j in cum_num]
				extractive_pred = extractive_pred[:2] if len(extractive_pred) > 0 else [0]
				prediction += json.dumps({"id":id[idx], "predict_sentence_index": extractive_pred[:2]}) + '\n'
			print(id[0], end ='\r')
		return prediction

def create_mini_batch(samples):	
	# 測試集有 labels
	tokens_tensors,  labels = zip(*samples)
	# zero pad 到同一序列長度
	try:
		tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
		labels = pad_sequence(labels, batch_first=True)
		# print(len(tokens_tensors), len(labels))
	except :
		return [],[]
	return tokens_tensors, labels

def create_mini_batch_test(samples):
	tokens_tensors,  words, words_pos, sentence = zip(*samples)
	try:
		tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
	except :
		return [], [], [], []
	return tokens_tensors, words, words_pos, sentence


def glove():
	embeddings_dict = {}
	with open("./glove.6B/glove.6B.300d.txt", 'r') as f:
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:], "float32")
			embeddings_dict[word] = vector
	return embeddings_dict


def main(argv, arc):
	test_path = argv[1]
	output_path = argv[2]
	test = pd.read_json(test_path, lines= True)
	
	glove_dict = glove()
	# nlp = en_core_web_sm.load()
	nlp = RegexpTokenizer(r'\w+')

	test_dataset = SummaryDataset(test, nlp, glove_dict, test = True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=create_mini_batch_test)

	with open(f'./model/model_extractive_0326_2300.pkl', 'rb') as inputs:
		model = pickle.load(inputs)
	prediction = model.test(test_loader)

	with open(f'{output_path}','w') as f:
		f.write(prediction)
if __name__ == '__main__':
	main(sys.argv, len(sys.argv))
