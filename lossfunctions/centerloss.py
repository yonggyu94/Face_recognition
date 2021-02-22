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
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.num_classes))

    def forward(self, x, labels):
        '''
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        '''
        # compute the distance of (x-center)^2
        batch_size = x.size(0)

        selected_centers = self.centers[labels, :]

        a = torch.sum(torch.mul(x, x), dim=1, keepdim=True)
        b = torch.sum(torch.mul(torch.transpose(selected_centers, 0, 1), torch.transpose(selected_centers, 0, 1)),
                      dim=0, keepdim=True)

        # a = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        # b = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = a + b
        distmat = distmat + -2 * torch.matmul(selected_centers, torch.transpose(x, 0, 1))

        # # get one_hot matrix
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # classes = torch.arange(self.num_classes).long().to(device)
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # dist = []
        # for i in range(batch_size):
        #     value = distmat[i][mask[i]]
        #     value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #     dist.append(value)
        # dist = torch.cat(dist)
        # loss = dist.mean()

        identity_matrix = torch.eye(batch_size).to(device)
        distmat = distmat * identity_matrix
        loss = torch.sum(distmat)
        return loss