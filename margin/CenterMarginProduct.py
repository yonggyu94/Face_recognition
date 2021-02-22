#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: ArcMarginProduct.py
@time: 2018/12/25 9:13
@desc: additive angular margin for arcface/insightface
'''

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CenterMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False):
        super(CenterMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        batch_size = x.size(0)
        selected_centers = self.weight[label, :]

        a = torch.sum(torch.mul(x, x), dim=1, keepdim=True)
        b = torch.sum(torch.mul(torch.transpose(selected_centers, 0, 1), torch.transpose(selected_centers, 0, 1)),
                      dim=0, keepdim=True)

        distmat = a + b
        distmat = distmat + -2 * torch.matmul(selected_centers, torch.transpose(x, 0, 1))

        identity_matrix = torch.eye(batch_size).to(device)
        distmat = distmat * identity_matrix
        center_loss = torch.sum(distmat)

        output = F.linear(x, self.weight)
        return output, center_loss


if __name__ == '__main__':
    pass