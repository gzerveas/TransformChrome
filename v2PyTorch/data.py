import os
import collections
import pdb
import csv
import json
import math
from pdb import set_trace as stop
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.utils.data


def load_csv_data(filename, window_sz):
    with open(filename) as fi:
        csv_reader = csv.reader(fi)
        csv_data = list(csv_reader)
    
    ncols=(len(csv_data[0]))
    nrows=len(csv_data)
    ngenes = int(nrows/window_sz)
    nfeatures = ncols-1
    
    hm_array = torch.zeros(ngenes, window_sz, 5)
    expr_array = torch.zeros(ngenes, 1)
    gene_ids = [None]*ngenes
    
    for gene_idx, i in enumerate(list(range(0, nrows, window_sz))): # Iterating through csv indices by window_sz
        window_data = csv_data[i:i+window_sz]
        for hm_idx, col_idx in enumerate(list(range(2,7))): # HM Indices in csv_data
            count_list = list(map(lambda row : int(row[col_idx]), window_data))
            hm_array[gene_idx, :, hm_idx] = torch.tensor(count_list)
        expr_array[gene_idx] = torch.tensor(int(window_data[0][7])) # Binary Expression
        gene_ids[gene_idx] = str(window_data[0][0].split("_")[0]) # Gene Ids
    
    return [hm_array, expr_array, gene_ids]


class HMData(torch.utils.data.Dataset): # Dataset class for loading data
    def __init__(self, filepath, window_sz=100):
        hm_array, expr_array, gene_ids = load_csv_data(filepath, window_sz)
        self.hm_array = hm_array
        self.expr_array = expr_array
        self.gene_ids = gene_ids
    def __len__(self):
        return len(self.hm_array)
    def __getitem__(self, idx):
        hm_data = self.hm_array[idx]
        expr_label = self.expr_array[idx]
        gene_id = self.gene_ids[idx]
        return hm_data, expr_label, gene_id


def load_data(cell_type):
    filepath = os.path.join('data/all_cell_data/', cell_type, 'classification')
    train_data = HMData(os.path.join(filepath,'train.csv'))
    valid_data = HMData(os.path.join(filepath,'valid.csv'))
    test_data = HMData(os.path.join(filepath,'test.csv'))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
    return train_loader, valid_loader, test_loader

def load_all_data():
    train_data_list = []
    val_data_list = []
    test_data_list = []
    cell_types = os.listdir('data/all_cell_data')
    for cell in cell_types:
        filepath = os.path.join('data/all_cell_data/', cell, 'classification')
        train_data = HMData(os.path.join(filepath,'train.csv'))
        valid_data = HMData(os.path.join(filepath,'valid.csv'))
        test_data = HMData(os.path.join(filepath,'test.csv'))
        train_data_list.append(train_data)
        val_data_list.append(val_data)
        test_data_list.append(test_data)
    
    train_data = torch.utils.data.ConcatDataset(train_data_list)
    valid_data = torch.utils.data.ConcatDataset(val_data_list)
    test_data = torch.utils.data.ConcatDataset(test_data_list)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
    return train_loader, valid_loader, test_loader

# def loadData(filename,windows):
#     with open(filename) as fi:
#         csv_reader=csv.reader(fi)
#         data=list(csv_reader)

#         ncols=(len(data[0]))
#     fi.close()
#     nrows=len(data)
#     ngenes=nrows/windows
#     nfeatures=ncols-1
#     print("Number of genes: %d" % ngenes)
#     print("Number of entries: %d" % nrows)
#     print("Number of HMs: %d" % nfeatures)

#     count=0
#     attr=collections.OrderedDict()

#     for i in range(0,nrows,windows):
#         hm1=torch.zeros(windows,1)
#         hm2=torch.zeros(windows,1)
#         hm3=torch.zeros(windows,1)
#         hm4=torch.zeros(windows,1)
#         hm5=torch.zeros(windows,1)
#         for w in range(0,windows):
#             hm1[w][0]=int(data[i+w][2])
#             hm2[w][0]=int(data[i+w][3])
#             hm3[w][0]=int(data[i+w][4])
#             hm4[w][0]=int(data[i+w][5])
#             hm5[w][0]=int(data[i+w][6])
#         geneID=str(data[i][0].split("_")[0])

#         thresholded_expr = int(data[i+w][7])

#         attr[count]={
#             'geneID':geneID,
#             'expr':thresholded_expr,
#             'hm1':hm1,
#             'hm2':hm2,
#             'hm3':hm3,
#             'hm4':hm4,
#             'hm5':hm5
#         }
#         count+=1

#     return attr


# class HMData(Dataset):
#     # Dataset class for loading data
#     def __init__(self,data_cell1,transform=None):
#         self.c1=data_cell1
#     def __len__(self):
#         return len(self.c1)
#     def __getitem__(self,i):
#         final_data_c1=torch.cat((self.c1[i]['hm1'],self.c1[i]['hm2'],self.c1[i]['hm3'],self.c1[i]['hm4'],self.c1[i]['hm5']),1)
#         label=self.c1[i]['expr']
#         geneID=self.c1[i]['geneID']
#         sample={'geneID':geneID,
#                'input':final_data_c1,
#                'label':label,
#                }
#         return sample

# def load_data(args):
#     '''
#     Loads data into a 3D tensor for each of the 3 splits.

#     '''
#     print("==>loading train data from: {}".format(args.train_file))
#     cell_train_dict1=loadData(args.train_file, args.n_bins)
#     train_inputs = HMData(cell_train_dict1)
#     Train = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True)

#     if args.valid_file is not None:
#         print("==>loading valid data from: {}".format(args.valid_file))
#         cell_valid_dict1=loadData(args.valid_file,args.n_bins)
#         valid_inputs = HMData(cell_valid_dict1)
#         Valid = torch.utils.data.DataLoader(valid_inputs, batch_size=args.batch_size, shuffle=False)
#     else:
#         Valid = None

#     if args.test_file is not None:
#         print("==>loading test data from: {}".format(args.test_file))
#         cell_test_dict1=loadData(args.test_file, args.n_bins)
#         test_inputs = HMData(cell_test_dict1)
#         Test = torch.utils.data.DataLoader(test_inputs, batch_size=args.batch_size, shuffle=False)
#     else:
#         Test = None

#     return Train, Valid, Test


