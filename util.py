# -*- coding:utf-8 -*-
"""
@Time：2023/03/27 18:01
@Author：KI
@File：util.py
@Motto：Hungry And Humble
"""
import copy
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from math import atan

torch.set_printoptions(profile="full")

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, add_self_loops


def motif_process(edge_index, num_nodes, path):
    edge_index = edge_index.cpu().numpy().tolist()
    write_edgelist = open(path + "edgelist.txt", 'w')

    for i in range(len(edge_index[0])):
        print(str(edge_index[0][i]) + " " + str(edge_index[1][i]), file=write_edgelist)

    write_edgelist.close()

    # transform edgelist to motif edgelist
    # os.system("models/pgd -f " + "data/edgelist.txt" + " -o natural --micro " + "data/motif_list.txt ")
    os.system("models/pgd -f " + "data/edgelist.txt" + " -o natural --micro " + "data/motif_list.txt > " + "data/motif_list_process.txt")
    #os.system("data/edgelist.txt" + " -o natural --micro " + "data/motif_list.txt > " + "data/motif_list_process.txt")
    # print('hhh')
    # load all motif edge - type:numpy
    motif_data = np.loadtxt(path + "motif_list.txt", skiprows=1, dtype=int, delimiter=',')
    motif_data = motif_data.T

    # renumber
    edge_data = np.loadtxt(path + "edgelist.txt", dtype=int).flatten()
    motif_data = renumber(edge_data, motif_data)

    motif_data = torch.from_numpy(motif_data)

    motif_data.edge_index = motif_data[0:2]
    motif_data.edge_attr = motif_data[2:]

    motif_edge_attr = []
    for m in range(len(motif_data.edge_attr)):
        index, attr = to_undirected(motif_data.edge_index, motif_data.edge_attr[m], num_nodes)
        attr = attr.float()
        if torch.abs(attr).sum(dim=0) > 0:
            histogram = plt.hist(attr.cpu().numpy().tolist(), density=True, log=False)
            plt.savefig('data/motif_%s.jpg' % m)
            plt.clf()

            attr = torch.cat((attr, torch.ones(num_nodes)), 0)
            one = torch.ones_like(attr)
            attr = torch.where(attr > 0, one, attr)

            motif_edge_attr.append(attr)

    motif_edge_attr = torch.stack(motif_edge_attr, 0)

    index_ = add_self_loops(index, num_nodes=num_nodes)
    motif_edge_index = index_[0]

    torch.save(motif_edge_attr, path + 'processed.pt')
    torch.save(motif_edge_index, path + 'index.pt')

    return motif_edge_index, motif_edge_attr


def renumber(edge_data, mo_data):
    order = dict()
    k = 0
    for i in range(len(edge_data)):
        if edge_data[i] not in order.values():
            k += 1
            order[k] = edge_data[i]

    for m in range(mo_data.shape[1]):

        if mo_data[0][m] in order.keys():
            mo_data[0][m] = order[mo_data[0][m]]

        if mo_data[1][m] in order.keys():
            mo_data[1][m] = order[mo_data[1][m]]

    return mo_data
