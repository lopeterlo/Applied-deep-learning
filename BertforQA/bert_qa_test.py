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

def clean_test(context, question):
	q_len = len(question)
	if q_len > 40:
		question = question[-40:]
		q_len = 40
	if len(context) + q_len + 3 < 512:
		return context, question
	else:
		remain = 512 - 3 - q_len
		return context[:remain], question
	
def load_data_to_df_test(test):
#   punc = '[，＠＃＄％＾＆＊，。、“\/.……%$!@#$「」：；%^&*()《》_+？—]'
	para_id, context, question, q_id, = [],[],[],[]
	for i in range(len(test['data'])):
		for article in test['data'][i]['paragraphs']:
			for qa in article['qas']:
				temp_context, temp_question = clean_test(article['context'], qa['question'])
				context.append(temp_context)
				question.append(temp_question)
				para_id.append(test['data'][i]['id'])
				q_id.append(qa['id'])

	data = {
		'para_id': para_id,
		'context': context,
		'question': question,
		'q_id': q_id,
	}
	test_df = pd.DataFrame(data=data)
	return test_df


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
#    start_pos, end_pos, answerable = self.reassign(inputs_ids, answer, )
#    print(answer)
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


def test(data, model, tokenizer, args):
	inputs_ids, token_type_ids, attention_mask = [i.cuda(args.device) if args.gpu else i for i in data[:3]]
	q_id = data[3]
	(start_score, end_score, answerable) = model(inputs_ids, attention_mask= attention_mask, token_type_ids= token_type_ids, args = args)
	a_val, a_idx = answerable.max(-1)

	start_score = start_score.masked_fill(token_type_ids == 0 ,  -100)
	end_score = end_score.masked_fill(token_type_ids == 0 ,  -100)
	s_val, s_idx = start_score.max(-1)
	e_val, e_idx = end_score.max(-1)
	ret = dict()
	for i in range(len(s_idx)):
		if a_idx[i] == 0:
			ans = ''
		else:
			ans = ''
			if s_idx[i] <= e_idx[i]:
				ans = inputs_ids[i][s_idx[i]: e_idx[i] + 1].tolist()
				ans = ''.join([tokenizer.convert_ids_to_tokens(i) for i in ans])
				spe_tok = ['#', '[CLS]', '[SEP]', '[UNK]', ' ']
				for j in spe_tok:
					ans = ans.replace(j, '')
		ret[q_id[i]] = ans[:30]
	return ret


def test_iters(model_path, test_path):
	args = Args_test()
	
	with open(test_path , 'rb') as f:
		test_data = json.loads(f.read())
	test_df = load_data_to_df_test(test_data)
	
	tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
	testset = QADataset(test_df, "test", tokenizer=tokenizer)
	testloader = DataLoader(testset, batch_size = args.batch_size)
	
	# load model
	model = BertQA.from_pretrained('bert-base-chinese')
	ckpt = torch.load(model_path)
	model.load_state_dict(ckpt['model_state_dict'])
	if args.gpu:
		model = model.cuda(args.device)
	prediction = dict()
	index = 0
	with Timer():
		with torch.no_grad():
			for data in testloader:
				pred = test(data, model, tokenizer, args)
				prediction.update(pred)
				print(f'prediction iterations: {index}, ans :{pred}', end = '\r')
				index += 1
	return prediction  



class Args_test():
	def __init__(self):
		self.batch_size = 5
		self.gpu = torch.cuda.is_available()
		self.device = 0


def main(argv, arc):
	test_path = argv[1]
	model_path = argv[2]
	prediction = test_iters(model_path, test_path)
	with open('prediction.json','w') as f:
		f.write(json.dumps(prediction, ensure_ascii=False))   
if __name__ == '__main__':
	main(sys.argv, len(sys.argv))