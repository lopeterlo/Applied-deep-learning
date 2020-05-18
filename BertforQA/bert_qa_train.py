import pandas as pd
import time
import math
import re
import json
import sys

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer, AdamW, BertPreTrainedModel
from transformers import get_linear_schedule_with_warmup


class Timer(object):
	""" A quick tic-toc timer
	Credit: http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
	"""
	def __init__(self, name=None, verbose=True):
		self.name = name
		self.verbose = verbose
		self.elapsed = None

	def __enter__(self):
		self.tstart = time.time()
		return self

	def __exit__(self, type, value, traceback):
		self.elapsed = time.time() - self.tstart
		if self.verbose:
			if self.name:
				print ('[%s]' % self.name,)
		print ('Total executing time for : %s' % self.elapsed)

def clean(context, answer, question, start_pos):
	q_len = len(question)
	if q_len > 40:
		question = question[-40:]
		q_len = 40
	if len(context) + q_len + 3 <= 512:
		return context, question
	else:
		remain = 512 - 3 - q_len
		if answer == '':
			return context[:remain], question
		if start_pos + len(answer) -1 < remain:
			return context[:remain], question
		else:
			return None, None

		
def load_data_to_df(data):
#	 punc = '[，＠＃＄％＾＆＊，。、“\/.……%$!@#$「」：；%^&*()《》_+？—]'
	para_id, context, question, q_id, answer, answer_start, answer_end, answerable, answer_start= [], [], [], [], [], [], [], [], []
	for i in range(len(data['data'])):
		for article in data['data'][i]['paragraphs']:
			for qa in article['qas']:
				start_pos = qa['answers'][0]['answer_start']
				temp_context, temp_question = clean(article['context'], qa['answers'][0]['text'], qa['question'], start_pos)
				if temp_context == None:
					continue
				answer_start.append(start_pos)
				question.append(temp_question)
				context.append(temp_context)
				answer.append(qa['answers'][0]['text'])
				para_id.append(data['data'][i]['id'])
				q_id.append(qa['id'])
				answerable.append(qa['answerable'])
	temp = {
		'para_id': para_id,
		'context': context,
		'question': question,
		'q_id': q_id,
		'answer': answer ,
		'answerable': answerable,
		'start_pos' : answer_start
	}
	df = pd.DataFrame(temp)
	return df


class QADataset(Dataset):
	def __init__(self, df, mode , tokenizer):
		assert mode in ["train", "test"]  # 
		self.mode = mode
		self.df = df
		self.tokenizer = tokenizer  # transformer 中的 BERT tokenizer
	
	def __getitem__(self, idx):
		data = self.df.iloc[idx]
		question = data['question']
		context = data['context']
		start_pos, end_pos = -1, -1
		if self.mode == 'train':
			encoded = self.tokenizer.encode_plus(
				question, 
				context, 
				pad_to_max_length = True,
				add_special_tokens  = True,
				return_tensors = 'pt',
				return_token_type_ids = True,
				return_attention_mask = True,
				truncation_strategy = 'only_second'
			)
			inputs_ids = encoded['input_ids'].squeeze(0)
			token_type_ids = encoded['token_type_ids'].squeeze(0)
			attention_mask = encoded['attention_mask'].squeeze(0)
			answer = data['answer']
			start_pos = data['start_pos']
			answerable = 1 if data['answerable'] else 0
			if answerable == 1:
#				 start_pos, end_pos, answerable = self.reassign(inputs_ids, answer, )
#				 print(answer)
				start_pos, end_pos, answerable = self.reassign(question, answer, context[:start_pos])
			start_pos, end_pos, answerable = torch.tensor(start_pos, dtype=torch.long),  torch.tensor(end_pos, dtype=torch.long), torch.tensor(answerable, dtype=torch.long)
			return (inputs_ids, token_type_ids, attention_mask, start_pos, end_pos, answerable)
		encoded = self.tokenizer.encode_plus(
				question, 
				context, 
				max_length  = 512,
				pad_to_max_length = True,
				add_special_tokens  = True,
				return_tensors = 'pt',
				return_token_type_ids = True,
				return_attention_mask = True,
				truncation_strategy = 'only_second'
			)
		
		inputs_ids = encoded['input_ids'][0]
		token_type_ids = encoded['token_type_ids'][0]
		attention_mask = encoded['attention_mask'][0]
		return (inputs_ids, token_type_ids, attention_mask, data['q_id'])
	
	
	def reassign(self, question, answer, b_context):
		q_len = len(self.tokenizer.encode(question, add_special_tokens = False, return_tensors ='pt')[0]) 
		b_len = len(self.tokenizer.encode(b_context, add_special_tokens = False, return_tensors ='pt')[0])
		ans_len = len(self.tokenizer.encode(answer, add_special_tokens = False, return_tensors ='pt')[0])
		start_pos = q_len + b_len + 2
		end_pos = start_pos + ans_len -1
		return start_pos, end_pos, 1

	
	def __len__(self):
		return len(self.df)


class BertQA(BertPreTrainedModel):
	def __init__(self, config):
		super(BertQA, self).__init__(config)
		self.num_labels = config.num_labels
		self.bert = BertModel(config)
		self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
		self.ans_outputs = nn.Linear(config.hidden_size, 2) # crossentropy
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.init_weights()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		start_positions=None,
		end_positions=None,
		answerable = None,
		args = None
	):

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		#outputs 
		
		sequence_output = outputs[0] # sequence_output : torch.Size([batch_size, 512, 768])
		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1) #torch.Size([batch_size, 512,])
		end_logits = end_logits.squeeze(-1)
		
		#answerable
		cls = outputs[1] # cls : torch.Size([batch_size, 768])
		cls = self.dropout(cls)
		ans_logits = self.ans_outputs(cls)

		outputs = (start_logits, end_logits, ans_logits) 
		if start_positions is not None and end_positions is not None :
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms

			ignored_index = start_logits.size(1)
			start_positions.clamp_(0, ignored_index)
			end_positions.clamp_(0, ignored_index)
			start_logits = start_logits.masked_fill(token_type_ids == 0 ,  -math.inf)
			end_logits = end_logits.masked_fill(token_type_ids == 0 , -math.inf)

			loss_fct = CrossEntropyLoss(ignore_index = 0)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			
			weight = torch.tensor([0.7, 0.3]).cuda(args.device)
			loss_fct1 =  CrossEntropyLoss(weight = weight)
			ans_loss = loss_fct1(ans_logits, answerable)
			
			total_loss = (start_loss + end_loss + ans_loss) / 3
			outputs = (total_loss, start_loss+ end_loss, ans_loss) + outputs

		   
		return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


def get_optimizer(model, trainloader_len, epoch, lr):
	optimizer = AdamW(model.parameters(), lr = lr)
	total_steps = trainloader_len * epoch
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 100, num_training_steps = total_steps)
	return optimizer, scheduler


def train(data, model, optimizer, scheduler, args):
	inputs_ids, token_type_ids, attention_mask, start_pos, end_pos, answerable = [i.cuda(args.device) if args.gpu else i for i in data ]
	outputs = model(inputs_ids, attention_mask= attention_mask, token_type_ids= token_type_ids ,start_positions = start_pos, end_positions = end_pos, answerable = answerable, args = args)
	loss, SE_loss, ans_loss = outputs [0], outputs[1], outputs[2]
	loss.backward() # calculate gradient
	torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
	optimizer.step()
	scheduler.step()
	model.zero_grad()
	return loss, SE_loss, ans_loss


def train_iters(train_path, val_path):
	with open(train_path  , 'rb') as f:
		train_data = json.loads(f.read())
	with open(val_path , 'rb') as f:
		valid_data = json.loads(f.read())
	train_df = load_data_to_df(train_data)
	valid_df = load_data_to_df(valid_data)
	args = Args()
	tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

	trainset = QADataset(train_df, "train", tokenizer=tokenizer)
	trainloader = DataLoader(trainset, batch_size = args.batch_size, drop_last = True, shuffle= True)

	validset = QADataset(valid_df, "train", tokenizer=tokenizer)
	validloader = DataLoader(validset, batch_size = args.batch_size, drop_last = True, shuffle= True)
	model = BertQA.from_pretrained('bert-base-chinese')
	if args.gpu:
		model = model.cuda(args.device)
	model.train()
#	 model.bert.embeddings.requires_grad = False
	train_loss_list, val_loss_list,  min_val_loss = [], [], 100000000
	for i in range(args.epoch):
		index, total, total_SE_loss, total_ans_loss, total_loss, val_loss, = 0, 0, 0, 0, 0, 0,
		optimizer, scheduler = get_optimizer(model, len(trainloader), args.epoch, args.lr)
		start_time = time.time()
		with Timer():
			for data in trainloader:
				loss, SE_loss, ans_loss = train(data, model, optimizer, scheduler, args)
				total += 1
				total_SE_loss += SE_loss
				total_ans_loss += ans_loss
				total_loss += loss 
				print(f'Epoch : { i + 1} / { args.epoch } ,iterations: {index}, SE_Loss: {total_SE_loss/ total}, ANS_Loss :{total_ans_loss/total},Average_Training Loss : {total_loss / total }', end = '\r')
				index += 1
			print(f'\n Total training time: {time.time() - start_time} ', end = '\n')
		train_loss_list.append(total_loss/ total)
		
		index, total, total_loss ,index = 0, 0, 0, 0
		start_time = time.time()
		with Timer():
			for data in validloader:
				loss, SE_loss, ans_loss = evals(data, model, args)
				total += 1
				total_loss += loss 
				print(f'Epoch : { i + 1} / { args.epoch } ,iterations: {index}, Average_Training Loss : {total_loss / total }', end = '\r')
				index += 1
			print(f'\n Total Validation time: {time.time() - start_time } ', end = '\n')
			val_loss = total_loss / total
			val_loss_list.append(val_loss)
		if val_loss < min_val_loss: 
			min_val_loss = val_loss
			checkpoint_path = f'./model/bertqa_ckpt_{i}.pt'
			torch.save({
					'epoch': i,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'train_loss_list' : train_loss_list,
					}, checkpoint_path)
	return train_loss_list, val_loss_list


def evals(data, model, args):
	with torch.no_grad():
		inputs_ids, token_type_ids, attention_mask, start_pos, end_pos, answerable = [i.cuda(args.device) if args.gpu else i for i in data ]
		outputs = model(inputs_ids, attention_mask= attention_mask, token_type_ids= token_type_ids ,start_positions = start_pos, end_positions = end_pos, answerable = answerable, args =args)
		loss, SE_loss, ans_loss = outputs [0], outputs[1], outputs[2]
	return loss, SE_loss, ans_loss


class Args():
	def __init__(self):
		self.epoch = 4
		self.lr = 1e-5
		self.batch_size = 10
		self.gpu = torch.cuda.is_available()
		self.device = 0


def main(argv, arc):
	train_path = argv[1]
	val_path = argv[2]
	train_iters(train_path, val_path)

if __name__ == '__main__':
	main(sys.argv, len(sys.argv))