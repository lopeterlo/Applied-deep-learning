import sys
import json
import spacy
import en_core_web_sm
import numpy as np
import random
import pickle
from collections import defaultdict
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import time

import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable


batch_size = 50
device = 1



class SummaryDataset(Dataset):
	def __init__(self, data, dic, test = False):
		self.data = data
		self.dic = dic
		self.test = test
		self.tokenizer = RegexpTokenizer(r'\w+')
		self.test = test
		
	def __len__(self):
		return len(self.data)
	 
	def __getitem__(self, idx):
		text = '_sos_ ' + self.data.loc[idx, 'text'] + ' _eos_'
		text_emb, text_word = self.get_emb(text)
		if not self.test:
			summary = self.data.loc[idx, 'summary'] + ' _eos_'
			summary_emb, summary_word = self.get_emb(summary)
			summary_word_id = self.get_summary_id_list(summary_word)
			length = len(summary_word_id)
			return torch.tensor(text_emb), length, torch.tensor(summary_word_id)
		id = self.data.loc[idx, 'id']
		return torch.tensor(text_emb), text_word, id

	def get_emb(self, data):
		tokens = self.tokenizer.tokenize(data)
		embeddings = []
		words = []
		for idx, token in enumerate(tokens):
			token = token.lower()
			emb = self.dic.get_emb(token)
			embeddings.append(emb)
			words.append(token)
		if len(embeddings) ==0:
			return [[0.0 for i in range(300)]], [[]]
		return embeddings, words
	 
	def get_summary_id_list(self, words):
		ans = []
		for i in words:
			word_id = self.dic.get_word_id(i)
			if word_id != -1:
				 ans.append(word_id)
		return ans

class words_dict():
	def __init__(self):
		self.word_count = {}
		self.id_to_word = {0: '_sos_', 1: '_eos_', 2: '_unk_'}
		self.word_to_id = {'_sos_': 0, '_eos_': 1, '_unk_': 2}
		self.n_words = 3
		self.tokenizer = RegexpTokenizer(r'\w+')
		self.remain_id = []
		self.glove = self.get_glove()
	
	def get_glove(self):
		embeddings_dict = {}
		with open("./glove.6B/glove.6B.300d.txt", 'r') as f:
			for line in f:
				values = line.split()
				word = values[0]
				vector = np.asarray(values[1:], "float32")
				embeddings_dict[word] = vector

		# add SOS, EOS and UNK
		embeddings_dict['_sos_'] = np.random.rand(300, )
		embeddings_dict['_eos_'] = np.random.rand(300, )
		embeddings_dict['_unk_'] = np.random.rand(300, )
		return embeddings_dict

	def add_word(self, sentence):
		tokens = self.tokenizer.tokenize(sentence)
		for token in tokens:
			token = token.lower()
			if token in self.glove.keys():
				if not self.word_to_id.get(token) :
					self.word_to_id[token] = self.n_words
					self.id_to_word[self.n_words] = token
					self.n_words += 1
					self.word_count[token] = 1
				else:
					self.word_count[token] += 1
						
	def reduce_dict(self, remain_ratio = 0.3):
		self.remain_id += [0,1,2]# SOS, EOS, UNK
		sort_d = sorted(self.word_count.items(), key = lambda x: x[1], reverse = True)[:int(self.n_words * remain_ratio)]
		for (i, j) in sort_d:
			self.remain_id.append(self.word_to_id[i])
		self.reconstruct()
	 
	def reconstruct(self):
		# reconstruct dict
		n_words =3
		id_to_word = {0: '_sos_', 1: '_eos_', 2: '_unk_'}
		word_to_id = {'_sos_': 0, '_eos_': 1, '_unk_': 2}
		for i in self.remain_id:
			if not word_to_id.get(i):
				 word_to_id[self.id_to_word[i]] = n_words
				 id_to_word[n_words] = self.id_to_word[i]
				 n_words += 1
		self.n_words = n_words
		self.id_to_word = id_to_word
		self.word_to_id = word_to_id
		self.remain_id = [i for i in range(n_words)]
		

	def get_emb(self, data):
		if self.word_to_id.get(data, -1) != -1:
			if self.word_to_id[data] in self.remain_id:
				return self.glove[data]
		return self.glove['_unk_']
		

	def get_word_id(self, data):
		if data == []:
			return -1
		if self.word_to_id.get(data, -1) != -1:
			if self.word_to_id[data] in self.remain_id:
				return self.word_to_id[data]
		return 2
		

def get_dict(merge_df, remain_dict_rate):
	dictionary = words_dict()
	for i in range(len(merge_df)):
		text = merge_df.loc[i, 'text']
		# data = text + summary
		dictionary.add_word(text)
	dictionary.reduce_dict(remain_dict_rate)
	for i in range(len(merge_df)):
		summary = merge_df.loc[i, 'summary']
		dictionary.add_word(summary)
	return dictionary
	

def create_mini_batch(samples):
	text_emb, length, summary_word_id = zip(*samples)
	text_emb = pad_sequence(text_emb, batch_first=True)
	summary_word_id = pad_sequence(summary_word_id, batch_first=True, padding_value=1)
	return text_emb, length, summary_word_id

def create_mini_batch_test(samples):
	text_emb, text_word, id = zip(*samples)
	text_emb = pad_sequence(text_emb, batch_first=True)
	return text_emb, text_word, id


class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers =1, bidirectional=False, dropout = 0):
		super(EncoderRNN, self).__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
									 dropout= dropout, bidirectional=bidirectional, batch_first = True)
		if bidirectional :
			self.l1 = nn.Linear(2*hidden_size, hidden_size)
		else:
			self.l1 = nn.Linear(hidden_size, hidden_size)
		self.relu = nn.ReLU()
		self.tan = nn.Tanh()
		self.init_weights()
		self.bidirectional = bidirectional
		self.drop = nn.Dropout(dropout)

	def forward(self, x):
		self.lstm.flatten_parameters()
		out, (hn, cn) = self.lstm(x)# out: tensor of shape (batch_size, seq_length, hidden_size)
		if self.bidirectional:
			hn = torch.cat((hn[0], hn[1]), 1)
			hn = hn.unsqueeze(0)
		hn = self.tan(self.drop(self.l1(hn)))
		return hn

	def init_weights(self):
		for name, p in self.lstm.named_parameters():
			if 'weight' in name:
				nn.init.orthogonal_(p)
			elif 'bias' in name:
				nn.init.constant_(p, 0)
					 
class DecoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, word_size, num_layers =1, dropout = 0):
		super(DecoderRNN, self).__init__()
		
		self.l1 = nn.Linear(input_size + hidden_size, word_size)
		self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, num_layers,
									 dropout= dropout, batch_first = True)
		self.relu = nn.ReLU()
		self.init_weights()

	def forward(self, x, h = None, c= None):
		self.lstm.flatten_parameters()
		out, (hn, cn) = self.lstm(x, (h, c))# out: tensor of shape (batch_size, seq_length, hidden_size)
		return hn, cn
	 
	def predict(self, x):
		out = self.l1(x)
		val, idx = out.max(-1)
		return out, idx
	 
	def test(self, x):
		out = self.l1(x)
		return out
	 
	def init_weights(self):
		for name, p in self.lstm.named_parameters():
			if 'weight' in name:
				nn.init.orthogonal_(p)
			elif 'bias' in name:
				nn.init.constant_(p, 0)

class AutoEncoderRNN(nn.Module):
	def __init__(self, input_size , hidden_size, word_size, num_layers=1, bidirectional=False, dropout = 0):
		super(AutoEncoderRNN, self).__init__()
		self.encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional, dropout=dropout)
		self.decoder = DecoderRNN(input_size, hidden_size, word_size, dropout=dropout)
	

def testIters(train_path, valid_path, test_path, output_path):

	checkpoint = torch.load('./model/abstractive_ckpt.pt')

	remain_dict_rate = checkpoint['remain_dict_rate']
	train_df = pd.read_json(train_path, lines= True)
	valid_df = pd.read_json(valid_path, lines= True)
	merge_df = train_df.append(valid_df, ignore_index= True)
	dictionary = get_dict(merge_df, remain_dict_rate = remain_dict_rate )

	
	input_size = checkpoint['input_size']
	hidden_size = checkpoint['hidden_size']
	bidirectional = checkpoint['bidirectional']
	dropout = checkpoint['dropout']
	num_layers = checkpoint['num_layers']

	model = AutoEncoderRNN(input_size, hidden_size, dictionary.n_words, num_layers = num_layers, bidirectional=bidirectional, dropout = dropout)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.cuda(device)

	test_dataset = SummaryDataset(valid_df, dictionary, test = True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, collate_fn = create_mini_batch_test)

	prediction = ''
	iteration = 0
	with torch.no_grad():
		for text_emb, text_word, id in test_loader:
			text_emb = text_emb.float().cuda(device)
			prediction += test(text_emb, id, dictionary, model, hidden_size)
			print(f'Testing loop, Iteration: {iteration} / {len(test_loader)}', end = '\r')
			iteration += 1
	with open(output_path,'w') as f:
		f.write(prediction)




def test(text_emb, id, dictionary, model, hidden_size,thres_ratio = 0.3):
	prediction = ''
	context = model.encoder(text_emb)  #torch.Size([1, 5, 150])
	hn = context
	cn = Variable(torch.zeros(1, batch_size, hidden_size)).cuda(device)
	
	SOS =  torch.tensor([[dictionary.get_emb('_sos_') for i in range(batch_size)]]).float().cuda(device)
	inputs = torch.cat((context, SOS), 2) #torch.Size([1, 5, 450])
	inputs = inputs.permute(1,0,2)
	words = SOS

#	   stop criteria
	thres = len(text_emb[0]) * thres_ratio
	ans = [[] for i in range(batch_size)]
	index = 0
	while True:
		hn, cn = model.decoder(inputs, hn, cn)  # torch.Size([1, 5, 150])
		combined = torch.cat((context, hn), -1)
		values, predict = model.decoder.predict(combined)  #torch.Size([1, 20, 98862]) torch.Size([1, 20])
	
		# val, pred = values.topk(3)
#			 print(pred)
		for i in range(batch_size):
			pred_word = dictionary.id_to_word[predict.tolist()[0][i]]
			ans[i].append(pred_word)
		
		words = torch.tensor([dictionary.get_emb(ans[j][index]) for j in range(batch_size)]).float().cuda(device)
		words = words.unsqueeze(0)
		
		inputs = torch.cat((hn, words), -1).permute(1,0,2)
#			 inputs = words.permute(1,0,2)
		index += 1
		
		#if predict summary exceed 40 words then stop
		if index >= thres:
			break

	for idx in range(batch_size):
		try:
			eos_idx = ans[idx].index('_eos_') + 1
			ans[idx] =  ans[idx][:eos_idx]
			prediction += json.dumps({"id":str(id[idx]), "predict": ' '.join(ans[idx])}) + '\n'
		except:
			prediction += json.dumps({"id":str(id[idx]), "predict": ' '.join(ans[idx])}) + '\n'
	return prediction
	

def main(argv, arc):
	train_path = argv[1]
	valid_path = argv[2]
	test_path = argv[3]
	output_path = argv[4]
	testIters(train_path, valid_path, test_path, output_path)
	

if __name__ == '__main__':
	main(sys.argv, len(sys.argv))
