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