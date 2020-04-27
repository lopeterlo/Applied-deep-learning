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
import re

import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable


input_size = 300
hidden_size = 300
batch_size = 32
lr = 1e-3
epoch = 5
teacher_forcing = 0.5
num_layers = 1
bidirectional = True
dropout = 0
device = 1
remain_dict_rate = 0.3


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
		data = data.replace("-",' ') 
		data = re.sub(r'[^a-zA-Z0-9. ]','',data)
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
		sentence = sentence.replace("-",' ') 
		sentence = re.sub(r'[^a-zA-Z0-9. ]','',sentence)
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
						
	def reduce_dict(self, remain_ratio = 0.5):
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
		summary = merge_df.loc[i, 'summary']
		data = text + summary
		dictionary.add_word(data)
	dictionary.reduce_dict(remain_dict_rate)
	return dictionary
	# for i in range(len(merge_df)):
	# 	summary = merge_df.loc[i, 'summary']
	# 	dictionary.add_word(summary)
	# return dictionary
	

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
			self.l2 = nn.Linear(2*input_size, input_size)
		else:
			self.l1 = nn.Linear(hidden_size, hidden_size)
			self.l2 = nn.Linear(input_size, input_size)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.init_weights()
		self.bidirectional = bidirectional
		self.drop = nn.Dropout(dropout)
		self.num_layers = num_layers

	def forward(self, x):
		self.lstm.flatten_parameters()
		out, (hn, cn) = self.lstm(x)# out: tensor of shape (batch_size, seq_length, hidden_size)
		if self.bidirectional:
			hn = torch.cat((hn[0], hn[1]), 1)
			hn = hn.unsqueeze(0)
		if self.num_layers > 1:
			hn = hn[0].unsqueeze(0)
		hn = self.tanh(self.drop(self.l1(hn)))
		out = self.tanh(self.drop(self.l1(out)))
		return out, hn

	def init_weights(self):
		for name, p in self.lstm.named_parameters():
			if 'weight' in name:
				nn.init.orthogonal_(p)
			elif 'bias' in name:
				nn.init.constant_(p, 0)

class Attention(nn.Module):
	def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional = False):
		super().__init__()
		self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
		self.v = nn.Linear(dec_hid_dim, 1, bias = False)
		
	def forward(self, hidden, encoder_outputs):
		#hidden = [batch size, 1, dec hid dim]
		#encoder_outputs = [batch size, src len, hidden_size]

		batch_size = encoder_outputs.shape[0]
		src_len = encoder_outputs.shape[1]
		
		#repeat decoder hidden state src_len times

		hidden = hidden.repeat(1, src_len, 1)  #hidden = [batch size, src len, dec hid dim]
		#encoder_outputs = [batch size, src len, enc hid dim * 2]
		energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), -1))) 
		
		#energy = [batch size, src len, dec hid dim]

		attention = self.v(energy).squeeze(2)
		
		#attention= [batch size, src len]
		return Func.softmax(attention, dim=1)
					 
class DecoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, word_size, num_layers =1, dropout = 0):
		super(DecoderRNN, self).__init__()

		self.l1 = nn.Linear(input_size + hidden_size, word_size)
		self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, num_layers,
									 dropout= dropout, batch_first = True)
		self.relu = nn.ReLU()
		self.init_weights()
		self.attention = Attention(hidden_size, hidden_size)

	def forward(self, x, h = None, c= None):
		self.lstm.flatten_parameters()
		out, (hn, cn) = self.lstm(x, (h, c))# out: tensor of shape (batch_size, seq_length, hidden_size)
		return hn, cn

	def predict(self, x, enc_outputs):
		# x is hidden size
		x = x.permute(1, 0, 2) # torch.Size([B, 1, 300])
		weight = self.attention(x, enc_outputs)
		weight = weight.unsqueeze(1)
		context = torch.bmm(weight, enc_outputs) # torch.Size([B, 1, 300])
		concat = torch.cat((context, x), -1) # torch.Size([B, 1, 600])
		concat = concat.permute(1,0,2) # torch.Size([1, 16, 600])
		out = self.l1(concat)
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
	def __init__(self, input_size, hidden_size, word_size, num_layers=1, bidirectional=False, dropout = 0):
		super(AutoEncoderRNN, self).__init__()
		self.encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional, dropout=dropout)
		self.decoder = DecoderRNN(input_size, hidden_size, word_size, dropout=dropout)


def trainIters(train_path, valid_path):
	train_df = pd.read_json(train_path, lines= True)
	valid_df = pd.read_json(valid_path, lines= True)
	merge_df = train_df.append(valid_df, ignore_index= True)
	dictionary = get_dict(merge_df, remain_dict_rate)

	model = AutoEncoderRNN(input_size, hidden_size, dictionary.n_words, bidirectional=bidirectional, dropout = dropout).cuda(device)

	train_dataset = SummaryDataset(train_df, dictionary)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, collate_fn = create_mini_batch ,drop_last = True, shuffle= True)
	valid_dataset = SummaryDataset(valid_df, dictionary)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, collate_fn = create_mini_batch, drop_last = True, )

	train_loss_list = []
	val_loss_list = []

	min_loss = 100000
	for i in range(epoch):
		iteration = 0
		total_loss= 0
		total_words = 0
		opt = torch.optim.Adam(model.parameters(), lr=lr)
	#	 opt_e = torch.optim.Adam(model.encoder.parameters(), lr=lr)
	#	 opt_d = torch.optim.Adam(model.decoder.parameters(), lr=lr)
		loss_f = nn.CrossEntropyLoss()
		start_time = time.time()
		for text_emb, length, summary_word_id in train_loader:
			text_emb = text_emb.float().cuda(device)
			summary_word_id = summary_word_id.cuda(device)
			batch_loss = train(text_emb, length, summary_word_id, dictionary, model, opt, loss_f)
			total_words += sum(length)
			total_loss += batch_loss
			iteration += 1
			print(f' Epoch : {i} / {epoch}, Iteration: {iteration}, batch_loss : {batch_loss/sum(length)}, avg_loss: {total_loss/ total_words} ', end = "\r")
		train_loss_list.append(total_loss/total_words)
		print(f'\n Total training time: {time.time() - start_time} ', end = '\n')
		iteration = 0	
		total_loss= 0
		total_words = 0
		start_time = time.time()
		for text_emb, length, summary_word_id in valid_loader:
			text_emb = text_emb.float().cuda(device)
			summary_word_id = summary_word_id.cuda(device)
			batch_loss = eval(text_emb, length, summary_word_id, dictionary, model, opt, loss_f)
			total_words += sum(length)
			total_loss += batch_loss
			iteration += 1
			print(f' Validation Loop: Epoch : {i}, Iteration: {iteration}, batch_loss : {batch_loss/sum(length)}, avg_loss: {total_loss/ total_words}', end = "\r")
		print(f'\n Total Validation time: {time.time() - start_time } ', end = '\n')

		valid_loss = total_loss / total_words
		val_loss_list.append(valid_loss)

		if valid_loss < min_loss:
			print(f' Validation loss improve from {min_loss} to {valid_loss} ')
			min_loss = valid_loss
			best_model = model

			checkpoint_path = f'./model/attention_ckpt_{i}.pt'
			torch.save({
					'epoch': i,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': opt.state_dict(),
					'input_size': input_size,
					'hidden_size': hidden_size,
					'bidirectional': bidirectional,
					'dropout': dropout,
					'num_layers': num_layers,
					'loss': min_loss,
					'teacher_forcing': teacher_forcing,
					'remain_dict_rate' : remain_dict_rate,
					'train_loss_list' : train_loss_list,
					'val_loss_list' : val_loss_list
					}, checkpoint_path)
		else:
			checkpoint_path = f'./model/attention_ckpt_{i}.pt'
			torch.save({
					'epoch': i,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': opt.state_dict(),
					'input_size': input_size,
					'hidden_size': hidden_size,
					'bidirectional': bidirectional,
					'dropout': dropout,
					'num_layers': num_layers,
					'loss': min_loss,
					'teacher_forcing': teacher_forcing,
					'remain_dict_rate': remain_dict_rate,
					'train_loss_list' : train_loss_list,
					'val_loss_list' : val_loss_list
					}, checkpoint_path)
			print(f' Validation loss did not improve from original {min_loss} to {valid_loss} ')
			# break
		print(f'=============================')
		
def train(text_emb, length, summary_word_id, dictionary, model, opt, loss_f):
	batch_loss = 0
	enc_outputs, context = model.encoder(text_emb)  #torch.Size([1, 5, 150])

	hn = context
	cn = Variable(torch.zeros(1, batch_size, hidden_size)).cuda(device)
	
	# first input with SOS token
	SOS =  torch.tensor([[dictionary.get_emb('_sos_') for i in range(batch_size)]]).float().cuda(device)
	inputs = torch.cat((context, SOS), 2) #torch.Size([1, B, 450])
	inputs = inputs.permute(1,0,2)  # torch.Size([B, 1, 450])

	words = SOS
	index = 0
	thres = int(summary_word_id.shape[1])

	while True:
		hn, cn = model.decoder(inputs, hn, cn)  # torch.Size([1, 5, 150])
		values, predict = model.decoder.predict(hn, enc_outputs)  #torch.Size([1, 20, 98862]) torch.Size([1, 20])
		for j in range(batch_size):
			if length[j] >= index:
				labels = summary_word_id[:,index].long().cuda(device)
				loss = loss_f(values[0], labels)
				batch_loss += loss

		use_teacher_forcing = True if random.random() < teacher_forcing else False
		# reconstruct input
		if use_teacher_forcing :
			words = [dictionary.id_to_word[labels.tolist()[j]] for j in range(batch_size)]
			words = torch.tensor([dictionary.get_emb(words[j]) for j in range(len(words))]).float().cuda(device)
		else:
			words = [dictionary.id_to_word[predict.view(-1).tolist()[j]] for j in range(batch_size)]
			words = torch.tensor([dictionary.get_emb(words[j]) for j in range(len(words))]).float().cuda(device)

		words = words.unsqueeze(0)
		inputs = torch.cat((hn, words), -1).permute(1,0,2) # h[0] torch.Size([B, 98862])
#			 inputs = words.permute(1,0,2)
		index += 1
		
		#if predict summary exceed thres
		if index >= thres:
			break
	batch_loss.backward()
	opt.step()
	opt.zero_grad()
	return batch_loss
	
		
def eval(text_emb, length, summary_word_id, dictionary, model, opt, loss_f):
	with torch.no_grad():
		batch_loss = 0
		enc_outputs, context = model.encoder(text_emb)#torch.Size([1, 5, 150])
		hn = context
		cn = Variable(torch.zeros(1, batch_size, hidden_size)).cuda(device)
		
		# first input with SOS token
		SOS =  torch.tensor([[dictionary.get_emb('_sos_') for i in range(batch_size)]]).float().cuda(device)
		inputs = torch.cat((context, SOS), 2) #torch.Size([1, B, 450])
		inputs = inputs.permute(1,0,2)  # torch.Size([B, 1, 450])

		words = SOS
		index = 0
		thres = int(summary_word_id.shape[1])

		while True:
			hn, cn = model.decoder(inputs, hn, cn)  # torch.Size([1, 5, 150])
			values, predict = model.decoder.predict(hn, enc_outputs)  #torch.Size([1, 20, 98862]) torch.Size([1, 20])
			for j in range(batch_size):
				if length[j] >= index:
					labels = summary_word_id[:,index].long().cuda(device)
					loss = loss_f(values[0], labels)
					batch_loss += loss
			
			words = [dictionary.id_to_word[predict.view(-1).tolist()[j]] for j in range(batch_size)]
			words = torch.tensor([dictionary.get_emb(words[j]) for j in range(len(words))]).float().cuda(device)

			words = words.unsqueeze(0)
			inputs = torch.cat((hn, words), -1).permute(1,0,2) # h[0] torch.Size([B, 98862])
	#			 inputs = words.permute(1,0,2)
			index += 1
			
			#if predict summary exceed thres
			if index >= thres:
				break

	return batch_loss


def main(argv, arc):
	train_path = argv[1]
	valid_path = argv[2]
	trainIters(train_path, valid_path)
	

if __name__ == '__main__':
	main(sys.argv, len(sys.argv))
