# -*- coding:utf-8 -*-

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch_geometric
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch_geometric import loader
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import scipy.sparse as sp
import torch.nn.functional as F
#from torch_geometric_temporal import DynamicGraphTemporalSignalBatch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calc_corr(a, b):
    s1 = Series(a)
    s2 = Series(b)
    return s1.corr(s2)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def create_graph(file_name):
    data = pd.read_csv(file_name)
    data = data.values   # 340 3
    edge_index = data[:, :2].T  # 2, x
    #
    if int(np.min(edge_index)) != 0:
        nodes = list(set(edge_index.flatten()))
        nodes = [int(x) for x in nodes]
        nodes_dict = dict(zip(nodes, [x for x in range(len(nodes))]))
        edge_index = edge_index.T.tolist()
        edge_index = [[nodes_dict[int(x)], nodes_dict[int(y)]] for x, y in edge_index]
        edge_index = torch.LongTensor(edge_index).T

    edge_index = torch.LongTensor(edge_index)
    # print(torch.min(edge_index[0]), torch.min(edge_index[1]))   # 0 0
    edge_weight = torch.FloatTensor(data[:, 2])
    print('edge_index:',edge_index.shape)
    print('edge_weight:',edge_weight.shape)    
    return edge_index, edge_weight


def adj2coo(adj):
    # adj numpy
    edge_index_temp = sp.coo_matrix(adj)
    values = edge_index_temp.data
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)

    return edge_index


def nn_seq(args, ):
    # 3 (26208 358 1)
    # 4 (16992, 307, 3)
    # 7 (28224, 883, 1)
    # 8 (17856, 170, 3)
    seq_len, B, pred_step_size = args.seq_len, args.batch_size, args.output_size
    npz_path = args.file_path + 'pems0' + args.file_path[-2] + '.npz'
    data = np.load(npz_path)['data']

    # 所有数据只用前1000条数据
    data = data[:500, :, :]

    data = data[:, :, 0]  # length, num_nodes
    num_nodes = data.shape[1]
    args.num_nodes = num_nodes

    # split
    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    # normalization
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    edge_index, edge_weight = create_graph(file_name=args.file_path + 'distance.csv')

    def process(dataset, batch_size, step_size, shuffle):
        # length num_nodes
        dataset = dataset.tolist()
        seq = []
        for i in tqdm(range(0, len(dataset) - seq_len - pred_step_size, step_size)):
            train_seq = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(len(dataset[0])):  #
                    x.append(dataset[j][c])
                train_seq.append(x)
            # 下几个时刻的所有变量
            train_labels = []
            for j in range(len(dataset[0])):
                train_label = []
                for k in range(i + seq_len, i + seq_len + pred_step_size):
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)
            # tensor
            train_seq = torch.FloatTensor(train_seq)
            # print(train_seq.shape)   # 24 13
            train_labels = torch.FloatTensor(train_labels)
            seq.append((train_seq, train_labels))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=pred_step_size, shuffle=False)

    return Dtr, Val, Dte, scaler, edge_index


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset


# nn_seq(seq_len=24, B=125, pred_step_size=1)
