import warnings
warnings.filterwarnings("ignore")
import argparse
import json
# import matplotlib
# import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cuda
import sys, os
import random
import numpy as np
from sklearn import metrics
# from SiameseLoss import ContrastiveLoss
import gc
import csv
from pdb import set_trace as stop
import sklearn

import models
import evaluate
import data

# python train.py --experiment_name=Cell1 --model_type=attchrome --train_file=train.csv --valid_file=valid.csv --test_file=valid.csv --epochs=120 --save_root=Results/

all_cell_types = os.listdir('data/all_cell_data')

parser = argparse.ArgumentParser(description='TransformChrome')
# parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--model_type', choices=['mlp','transformer','atten_chrome'])
# parser.add_argument('--clip', type=float, default=1,help='gradient clipping')
# parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 16 default')
# parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout) if n_layers LSTM > 1')
# parser.add_argument('--experiment_name', type=str, default='my_exp', help='experiment name')
parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
parser.add_argument('--cell_type', choices=['all','individual'])
# parser.add_argument('--n_bins', type=int, default=100, help='number of bins')
# parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
# parser.add_argument('--save_attention_maps',action='store_true', help='set to save validation beta attention maps')
# parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attnetion maps')
# parser.add_argument('--test_saved_model',action='store_true', help='only test saved model')
args = parser.parse_args()

torch.manual_seed(1)

model_name = args.model_type

print('the model name: ', model_name)

model_dir = os.path.join(args.save_root, model_name)
print('saving results in: ', model_dir)
os.makedirs(model_dir, exist_ok=True)

output_csv_file_train = os.path.join(args.save_root, f'{model_name}_{args.cell_type}_train.csv')
counter = 0
while os.path.exists(output_csv_file_train):
	counter+=1
	output_csv_file_train = output_csv_file_train[:-4]+('-')+str(counter)+('.csv')

output_csv_file_val = os.path.join(args.save_root, f'{model_name}_{args.cell_type}_valid.csv')
counter = 0
while os.path.exists(output_csv_file_val):
	counter+=1
	output_csv_file_val = output_csv_file_val[:-4]+('-')+str(counter)+('.csv')

output_csv_file_test = os.path.join(args.save_root, f'{model_name}_{args.cell_type}_test.csv')
counter = 0
while os.path.exists(output_csv_file_test):
	counter+=1
	output_csv_file_test = output_csv_file_test[:-4]+('-')+str(counter)+('.csv')

def train(model, train_loader, n_epochs=1):
	model = model.train()
	per_epoch_loss = 0
	for epoch_idx in range(n_epochs):
		for idx, batch in enumerate(train_loader):
			hm_array, expr_label, _ = batch
			hm_array = hm_array.cuda()
			expr_label = expr_label.cuda()
			predictions = model(hm_array)
			loss = model.loss(predictions, expr_label)
			loss.backward()
			norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1)
			optimizer.step()
			optimizer.zero_grad()
			per_epoch_loss += loss.item()
			# if idx % 1000 == 0:
			# 	preds = predictions.detach().cpu()
			# 	targets = expr_label.detach().cpu()
			# 	train_avgAUPR, train_avgAUC = evaluate.compute_metrics(preds, targets)
			# 	print(f'Batch #{idx}- AUPR: {train_avgAUPR}, AUC: {train_avgAUC}')
		
		per_epoch_loss = per_epoch_loss/len(train_loader.dataset)
		print(f'Epoch #{epoch_idx+1}; Loss:{per_epoch_loss}')
		per_epoch_loss = 0
	
	return model

def get_threshold(fpr, cutoff=0.05):
	for i,x in enumerate(fpr):
		if x > cutoff:
			threshold_idx = i-1
			break
	
	return threshold_idx

def eval_model(model, eval_loader, mode='valid'):
	# Validation Testing
	model = model.eval()
	num_correct = 0
	total_number = eval_loader.dataset.__len__()
	val_loss = 0
	all_preds = []
	all_labels = []
	for idx, batch in enumerate(eval_loader):
		hm_array, expr_label, _ = batch
		hm_array = hm_array.cuda()
		expr_label = expr_label.cuda()
		predictions = model(hm_array)
		loss = model.loss(predictions, expr_label)
		val_loss += loss.item()
		
		# model_predictions = torch.sigmoid(predictions).detach().cpu().numpy()
		model_predictions = predictions.detach().cpu().numpy()
		all_preds.append(model_predictions)
		
		actual_labels = expr_label.detach().cpu().numpy()
		all_labels.append(actual_labels)
	
	all_preds = np.concatenate(all_preds, 0).squeeze(1)
	all_labels = np.concatenate(all_labels, 0).squeeze(1)
	all_labels = all_labels.astype(int)
	
	precision, recall, thresholds = metrics.precision_recall_curve(all_labels, all_preds, pos_label=1)
	auPR = sklearn.metrics.auc(recall, precision)
	auROC = sklearn.metrics.roc_auc_score(all_labels, all_preds)
	avg_precision = sklearn.metrics.average_precision_score(all_labels, all_preds)
	
	# fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_preds, pos_label=1)
	# threshold_idx = get_threshold(fpr)
	# threshold = thresholds[threshold_idx]
	# pred_labels = all_preds > threshold
	# f1_score = metrics.f1_score(all_labels, pred_labels)
	# all_metrics = metrics.classification_report(all_labels, pred_labels)
	
	# eval_metrics = {'AUROC' : auROC, 'AUPR' : auPR, 'AP': avg_precision}
	eval_metrics = [auROC, auPR, avg_precision]
	print(f'{mode}-  AUC: {auROC}, AUPR: {auPR}, AP: {avg_precision}')
	
	return eval_metrics

# cell_type = 'E084'
# cell_type = 'E003'
# cell_type = 'E116' # GM12878
# dataloaders = data.load_all_data()
def get_new_model():
	if args.model_type == 'atten_chrome':
		default_args = argparse.Namespace
		default_args.n_bins = 100
		default_args.batch_size = 16
		default_args.n_hms = 5
		default_args.num_layers = 1
		default_args.dropout = 0.5
		default_args.bidirectional = True
		default_args.bin_rnn_size = 32
		model = models.att_chrome(default_args)
	elif args.model_type == 'transformer':
		model = models.transformer_encoder()
	elif args.model_type == 'mlp':
		model = models.baseline_model()
	else:
		raise NotImplementedError
	
	return model

lr = 0.0001


cell_type_metrics = {}

if args.choices == 'all':
	dataloaders = data.load_all_data(args.batch_size)
	train_loader, val_loader, test_loader = dataloaders
	
	model = get_new_model()
	model = model.cuda()
	optimizer = optim.Adam(model.parameters(), lr = lr)
	model = train(model, train_loader, 5)
	
	for cell_type in all_cell_types:
		dataloaders = data.load_data(cell_type)
		train_loader, val_loader, test_loader = dataloaders
		
		train_metrics = eval_model(model, train_loader, 'train')
		val_metrics = eval_model(model, val_loader, 'valid')
		test_metrics = eval_model(model, test_loader, 'test')
		cell_type_metrics[cell_type] = [val_metrics, test_metrics]
	
elif args.choice == 'individual':
	for cell_type in all_cell_types:
		dataloaders = data.load_data(cell_type)
		train_loader, val_loader, test_loader = dataloaders
		
		model = get_new_model()
		model = model.cuda()
		optimizer = optim.Adam(model.parameters(), lr = lr)
		model = train(model, train_loader, 30)
		
		train_metrics = eval_model(model, train_loader, 'train')
		val_metrics = eval_model(model, val_loader, 'valid')
		test_metrics = eval_model(model, test_loader, 'test')
		cell_type_metrics[cell_type] = [train_metrics, val_metrics, test_metrics]
		print(f"Finished {cell_type}")
else:
	raise NotImplementedError


with open(output_csv_file_train, 'w') as f:
	for cell_type, metric_list in cell_type_metrics.items():
		metrics = [str(x)[:6] for x in metric_list[0]]
		line = ','.join([cell_type] + metrics) + '\n'
		f.write(line)
	f.close()

with open(output_csv_file_valid, 'w') as f:
	for cell_type, metric_list in cell_type_metrics.items():
		metrics = [str(x)[:6] for x in metric_list[0]]
		line = ','.join([cell_type] + metrics) + '\n'
		f.write(line)
	f.close()

with open(output_csv_file_test, 'w') as f:
	for cell_type, metric_list in cell_type_metrics.items():
		metrics = [str(x)[:6] for x in metric_list[0]]
		line = ','.join([cell_type] + metrics) + '\n'
		f.write(line)
	f.close()




# args_dict = {'lr': 0.0001, 'model_name': 'attchrome', 'clip': 1, 'epochs': 2, 'batch_size': 10, 'dropout': 0.5, 'cell_1': 'Cell1', 'save_root': 'Results/Cell1', 
#'data_root': 'data/', 'gpuid': 0, 'gpu': 0, 'n_hms': 5, 'n_bins': 200, 'bin_rnn_size': 32, 'num_layers': 1, 'unidirectional': False, 'save_attention_maps': False, 
#'attentionfilename': 'beta_attention.txt', 'test_on_saved_model': False, 'bidirectional': True, 'dataset': 'Cell1'}


# attentionmapfile = os.path.join(args.save_root, args.attentionfilename)
# print('==>processing data')



# print('==>building model')
# model = Model.att_chrome(args)


# if torch.cuda.device_count() > 0:
# 	torch.cuda.manual_seed_all(1)
# 	dtype = torch.cuda.FloatTensor
# 	# cuda.set_device(args.gpuid)
# 	model.type(dtype)
# 	print('Using GPU '+str(args.gpuid))
# else:
# 	dtype = torch.FloatTensor

# print(model)
# if(args.test_saved_model==False):
# 	print("==>initializing a new model")
# 	for p in model.parameters():
# 		p.data.uniform_(-0.1,0.1)


# optimizer = optim.Adam(model.parameters(), lr = args.lr)
# #optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
# def train(TrainData):
# 	model.train()
# 	# initialize attention
# 	diff_targets = torch.zeros(TrainData.dataset.__len__(),1)
# 	predictions = torch.zeros(diff_targets.size(0),1)

# 	all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(args.n_hms*args.n_bins))
# 	all_attention_hm=torch.zeros(TrainData.dataset.__len__(),args.n_hms)

# 	num_batches = int(math.ceil(TrainData.dataset.__len__()/float(args.batch_size)))
# 	all_gene_ids=[None]*TrainData.dataset.__len__()
# 	per_epoch_loss = 0
# 	print('Training')
# 	for idx, Sample in enumerate(TrainData):

# 		start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, TrainData.dataset.__len__())
	

# 		inputs_1 = Sample['input']
# 		batch_diff_targets = Sample['label'].unsqueeze(1).float()

		
# 		optimizer.zero_grad()
# 		batch_predictions= model(inputs_1.type(dtype))

# 		loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets,reduction='mean')

# 		per_epoch_loss += loss.item()
# 		loss.backward()
# 		torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
# 		optimizer.step()

# 		# all_attention_bin[start:end]=batch_alpha.data
# 		# all_attention_hm[start:end]=batch_beta.data

# 		diff_targets[start:end,0] = batch_diff_targets[:,0]
# 		all_gene_ids[start:end]=Sample['geneID']
# 		predictions[start:end] = batch_predictions.data.cpu()
		
# 	per_epoch_loss=per_epoch_loss/num_batches
# 	return predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids



# def test(ValidData,split_name):
# 	model.eval()

# 	diff_targets = torch.zeros(ValidData.dataset.__len__(),1)
# 	predictions = torch.zeros(diff_targets.size(0),1)

# 	all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(args.n_hms*args.n_bins))
# 	all_attention_hm=torch.zeros(ValidData.dataset.__len__(),args.n_hms)

# 	num_batches = int(math.ceil(ValidData.dataset.__len__()/float(args.batch_size)))
# 	all_gene_ids=[None]*ValidData.dataset.__len__()
# 	per_epoch_loss = 0
# 	print(split_name)
# 	for idx, Sample in enumerate(ValidData):

# 		start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, ValidData.dataset.__len__())
# 		optimizer.zero_grad()

# 		inputs_1 = Sample['input']
# 		batch_diff_targets= Sample['label'].unsqueeze(1).float()

# 		# batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))
# 		batch_predictions = model(inputs_1.type(dtype))

# 		loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets,reduction='mean')
# 		# all_attention_bin[start:end]=batch_alpha.data
# 		# all_attention_hm[start:end]=batch_beta.data


# 		diff_targets[start:end,0] = batch_diff_targets[:,0]
# 		all_gene_ids[start:end]=Sample['geneID']
# 		predictions[start:end] = batch_predictions.data.cpu()

# 		per_epoch_loss += loss.item()
# 	per_epoch_loss=per_epoch_loss/num_batches
# 	return predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids




# best_valid_loss = 10000000000
# best_valid_avgAUPR=-1
# best_valid_avgAUC=-1
# best_test_avgAUC=-1
# best_train_avgAUC = -1
# if(args.test_saved_model==False):
# 	for epoch in range(0, args.epochs):
# 		print('---------------------------------------- Training '+str(epoch+1)+' -----------------------------------')
# 		predictions,diff_targets,alpha_train,beta_train,train_loss,_ = train(Train)
# 		train_avgAUPR, train_avgAUC = evaluate.compute_metrics(predictions,diff_targets)

# 		if Valid is not None:
# 			predictions,diff_targets,alpha_valid,beta_valid,valid_loss,gene_ids_valid = test(Valid,"Validation")
# 			valid_avgAUPR, valid_avgAUC = evaluate.compute_metrics(predictions,diff_targets)

# 		if Test is not None:
# 			predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test = test(Test,'Testing')
# 			test_avgAUPR, test_avgAUC = evaluate.compute_metrics(predictions,diff_targets)

# 		if Valid is not None:
# 			best_metric = best_valid_avgAUC
# 			current_metric = valid_avgAUC
# 		elif Test is not None:
# 			best_metric = best_test_avgAUC
# 			current_metric = test_avgAUC
# 		else:
# 			best_metric = best_train_avgAUC
# 			current_metric = train_avgAUC


# 		if current_metric > best_metric:
# 				# save best epoch -- models converge early
# 			if Valid is not None:
# 				best_valid_avgAUC = valid_avgAUC
# 			if Test is not None:
# 				best_test_avgAUC = test_avgAUC
# 			torch.save(model.cpu().state_dict(),model_dir+"/"+model_name+'_avgAUC_model.pt')
# 			model.type(dtype)

# 		print("Epoch:",epoch)
# 		print("approx. train avgAUC:",train_avgAUC)
# 		if Valid is not None:
# 			print("valid avgAUC:",valid_avgAUC)
# 			print("best valid avgAUC:", best_valid_avgAUC)
# 		if Test is not None:
# 			print("test avgAUC:",test_avgAUC)
# 			print("best test avgAUC:", best_test_avgAUC)

# 	print("\nFinished training")
# 	if Valid is not None:
# 		print("Best validation avgAUC:",best_valid_avgAUC)
# 	if Test is not None:
# 		print("Best test avgAUC:",best_test_avgAUC)



# 	if(args.save_attention_maps):
# 		attentionfile=open(attentionmapfile,'w')
# 		attentionfilewriter=csv.writer(attentionfile)
# 		beta_test=beta_test.numpy()
# 		for i in range(len(gene_ids_test)):
# 			gene_attention=[]
# 			gene_attention.append(gene_ids_test[i])
# 			for e in beta_test[i,:]:
# 				gene_attention.append(str(e))
# 			attentionfilewriter.writerow(gene_attention)
# 		attentionfile.close()


# else:
# 	model=torch.load(model_dir+"/"+model_name+'_avgAUC_model.pt')
# 	predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test = test(Test)
# 	test_avgAUPR, test_avgAUC = evaluate.compute_metrics(predictions,diff_targets)
# 	print("test avgAUC:",test_avgAUC)

# 	if(args.save_attention_maps):
# 		attentionfile=open(attentionmapfile,'w')
# 		attentionfilewriter=csv.writer(attentionfile)
# 		beta_test=beta_test.numpy()
# 		for i in range(len(gene_ids_test)):
# 			gene_attention=[]
# 			gene_attention.append(gene_ids_test[i])
# 			for e in beta_test[i,:]:
# 				gene_attention.append(str(e))
# 			attentionfilewriter.writerow(gene_attention)
# 		attentionfile.close()
