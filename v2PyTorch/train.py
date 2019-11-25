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


import models
import evaluate
import data

# python train.py --experiment_name=Cell1 --model_type=attchrome --train_file=train.csv --valid_file=valid.csv --test_file=valid.csv --epochs=120 --save_root=Results/

parser = argparse.ArgumentParser(description='DeepDiff')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--model_type', type=str, default='attchrome', help='DeepDiff variation')
parser.add_argument('--clip', type=float, default=1,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout) if n_layers LSTM > 1')
parser.add_argument('--experiment_name', type=str, default='my_exp', help='experiment name')
parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
parser.add_argument('--train_file', help='training set file (optional if `test_saved_model` is True)')
parser.add_argument('--valid_file', help='validation set file (optional)')
parser.add_argument('--test_file', help='test set file (optional)')
parser.add_argument('--n_bins', type=int, default=100, help='number of bins')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--save_attention_maps',action='store_true', help='set to save validation beta attention maps')
parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attnetion maps')
parser.add_argument('--test_saved_model',action='store_true', help='only test saved model')
args = parser.parse_args()

# torch.manual_seed(1)

model_name = (args.experiment_name)+('_')
model_name += args.model_type

print('the model name: ',model_name)

args.save_root = os.path.join(args.save_root, model_name)
print('saving results in: ', args.save_root)
model_dir = os.path.join(args.save_root, "checkpoints")
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

# cell_type = 'E084'
# cell_type = 'E003'
cell_type = 'E116' # GM12878
# dataloaders = data.load_all_data()
dataloaders = data.load_data(cell_type)
train_loader, val_loader, test_loader = dataloaders


lr = 0.0001
# model = models.transformer_encoder().cuda()
model = simple_model().cuda()
# model = models.att_chrome(args).cuda()
for p in model.parameters():
	p = p.data.uniform_(-0.1,0.1)

optimizer = optim.Adam(model.parameters(), lr = lr)

per_epoch_loss = 0
for epoch_idx in range(2):
	model = model.train()
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
		if idx % 1000 == 0:
			preds = predictions.detach().cpu()
			targets = expr_label.detach().cpu()
			train_avgAUPR, train_avgAUC = evaluate.compute_metrics(preds, targets)
			print(f'Batch #{idx}- AUPR: {train_avgAUPR}, AUC: {train_avgAUC}')
	
	per_epoch_loss = per_epoch_loss/len(train_loader.dataset)
	print(f'Epoch #{epoch_idx+1}; Loss:{per_epoch_loss}')
	per_epoch_loss = 0
	
	# Validation Testing
	model = model.eval()
	num_correct = 0
	total_number = val_loader.dataset.__len__()
	val_loss = 0
	all_preds = []
	all_labels = []
	for idx, batch in enumerate(val_loader):
		hm_array, expr_label, _ = batch
		hm_array = hm_array.cuda()
		expr_label = expr_label.cuda()
		predictions = model(hm_array)
		loss = model.loss(predictions, expr_label)
		val_loss += loss.item()
		
		# model_predictions = torch.sigmoid(predictions).detach().cpu().numpy()
		model_predictions = predictions.detach().cpu()
		all_preds.append(model_predictions)
		model_predictions = model_predictions.numpy()
		mean_prediction = model_predictions.mean()
		model_predictions = model_predictions > 0.5
		
		actual_labels = expr_label.detach().cpu()
		all_labels.append(actual_labels)
		actual_labels = actual_labels.numpy()
		actual_labels = actual_labels > 0.5
		matching_preds = np.logical_and(actual_labels, model_predictions)
		num_correct += np.count_nonzero(matching_preds)
	
	all_preds = torch.cat(all_preds, 0)
	all_labels = torch.cat(all_labels, 0)
	valid_avgAUPR, valid_avgAUC = evaluate.compute_metrics(all_preds, all_labels)
	print(f'Validation- AUPR: {valid_avgAUPR}, AUC: {valid_avgAUC}')
	val_loss = val_loss/total_number
	accuracy = 100*num_correct/total_number
	print(f'Val Loss: {val_loss}; Accuracy: {accuracy}')


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