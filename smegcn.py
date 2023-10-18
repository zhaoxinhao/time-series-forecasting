# -*- coding:utf-8 -*-
"""
@Time：2023/03/27 12:16
@Author：KI
@File：smegcn.py
@Motto：Hungry And Humble
"""
import warnings
warnings.filterwarnings('ignore')

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from math import sqrt
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.dense.linear import Linear
from .module import *


class SMEGCN(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[Tensor]

    def __init__(self, args, in_channels, out_channels, cached: bool = False, add_self_loops: bool = False,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SMEGCN, self).__init__(**kwargs)

        self.add_self_loops = False
        self.normalize = normalize
        self.motif_num = args.motif_num

        self._cached_edge_index = None
        self._cached_adj_t = None

        for k, v in vars(args).items():
            setattr(self, k, v)

        self.input_trans = torch.nn.Linear(in_channels, out_channels)
        self.output_trans = torch.nn.Linear(out_channels, out_channels)
        self.type_norm = 'batch'
        if self.type_norm == 'batch':
            self.input_bn = torch.nn.BatchNorm1d(out_channels)
            self.layers_bn = torch.nn.ModuleList([])
            for _ in range(self.sme_num_layers):
                self.layers_bn.append(torch.nn.BatchNorm1d(out_channels))

        self.propogation = self.message_passing

        if self.sme_num_layers <= self.early:
            self.early = self.sme_num_layers
            self.last_para = torch.nn.Parameter(
                torch.ones(self.early), requires_grad=True)
            torch.nn.init.uniform_(self.last_para, 0, 1)
        else:
            self.post_para = torch.nn.Parameter(torch.ones(self.motif_num * (self.sme_num_layers - self.early)))
            torch.nn.init.uniform_(self.post_para, 0, 1)
            self.post_bn = torch.nn.BatchNorm1d(out_channels)
            self.last_para = torch.nn.Parameter(torch.ones(self.early + 1), requires_grad=True)
            torch.nn.init.uniform_(self.last_para, 0, 1)

        self.reg_params = list(self.input_trans.parameters())
        self.non_reg_params = list(self.output_trans.parameters())

        if self.type_norm == 'batch':
            for bn in self.layers_bn:
                self.reg_params += list(bn.parameters())

        self.optimizer = torch.optim.Adam([
            dict(params=self.reg_params, weight_decay=self.weight_decay1),
            dict(params=self.non_reg_params, weight_decay=self.weight_decay2)
        ], lr=self.lr)

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def ifnormalize(self, x, edge_index, edge_weight):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(0), False,
                    self.add_self_loops, dtype=x.dtype)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(0), False,
                    self.add_self_loops, dtype=x.dtype)

        return edge_index, edge_weight

    def message_passing(self, x, edge_index, edge_weight, layers):

        layers_output = []
        for k in range(layers):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(
                        edge_weight, p=self.dropout, training=self.training)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout,
                                      training=self.training)
                    edge_index = edge_index.set_value(value, layout='coo')

            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)

            if self.type_norm == 'batch':
                x = self.layers_bn[k](x)
            layers_output.append(x)

        layers_output = torch.stack(layers_output, 0)

        return x, layers_output

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        x = self.input_trans(x)
        if self.type_norm == 'batch':
            x = self.input_bn(x)
        x = F.relu(x)  # predict after propogation
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        h = self.output_trans(x)
        x = h
        h0 = x

        x0 = x
        post = []

        if self.early > 0:
            e_i_0, e_w_0 = self.ifnormalize(x, edge_index, edge_weight[0])
            x0, layer0 = self.propogation(x0, e_i_0, e_w_0, self.early)  # layer0

        if self.sme_num_layers - self.early > 0:
            for edge_w in enumerate(edge_weight):
                e_i, e_w = self.ifnormalize(x, edge_index, edge_w[1])
                tmp, layers = self.propogation(x0, e_i, e_w, self.sme_num_layers - self.early)

                post.append(layers)

            if self.early == 0:
                layer0 = torch.zeros_like(layers)

            post = torch.cat(post, 0)

            post = post.permute(1, 2, 0)
            post_para = F.softmax(self.post_para)
            # print(post.shape) # 13 64 6
            # print(post_para.shape)  # 8
            post = post_para * post

            post = torch.sum(post, dim=2, keepdim=True)

            total = torch.cat((h0.unsqueeze(dim=2), layer0.permute(1, 2, 0)), dim=2)

        else:
            total = layer0.permute(1, 2, 0)

        w_aggr = F.softmax(self.last_para)
        x = w_aggr * total
        x = torch.sum(x, dim=2)
        post = torch.squeeze(post)
        x = (1 - self.alpha) * x + self.alpha * post

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)
