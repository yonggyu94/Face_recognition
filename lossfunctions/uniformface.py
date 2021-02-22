#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: centerloss.py
@time: 2019/1/4 15:24
@desc: the implementation of center loss
'''

import torch
import torch.nn.functional as F


def uniform(selected_centers, adjacency_not):
    batch_size = selected_centers.shape[0]
    selected_centers = F.normalize(selected_centers)
    a = torch.sum(torch.mul(selected_centers, selected_centers), dim=1, keepdim=True)
    b = torch.sum(torch.mul(torch.transpose(selected_centers, 0, 1), torch.transpose(selected_centers, 0, 1)),
                  dim=0, keepdim=True)
    ab = torch.matmul(selected_centers, torch.transpose(selected_centers, 0, 1))

    pd_mat = a + b - 2.0 * ab
    error_mask = pd_mat > 0.0
    pd_mat = pd_mat * error_mask.type(torch.cuda.FloatTensor)

    margin_mask = pd_mat <= 0.0
    # print(torch.sum(pd_mat).item())
    pd_mat2 = torch.sqrt(pd_mat + margin_mask.type(torch.cuda.FloatTensor) * 1e-16) + 1.0
    # print(torch.sum(pd_mat2).item())
    pd_mat2 = torch.mul(1.0 / pd_mat2, adjacency_not)
    # print(torch.sum(pd_mat2).item())

    uniform_loss = torch.sum(pd_mat2) / batch_size * (batch_size - 1.0)
    # print(uniform_loss)
    return uniform_loss