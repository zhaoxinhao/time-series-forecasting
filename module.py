# -*- coding:utf-8 -*-
"""
@Time：2023/03/27 12:15
@Author：KI
@File：module.py
@Motto：Hungry And Humble
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add


class Prop(MessagePassing):
    def __init__(self, num_classes, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.proj = Linear(num_classes, 1)

    def forward(self, preds):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)

        # pps = torch.stack(preds, dim=1)
        pps = preds
        # print(pps.shape)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
