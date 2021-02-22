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
import torch.nn.functional as F


def regularizer(weight, cls_num):
    weight_norm = F.normalize(weight)
    cos = torch.mm(weight_norm, weight_norm.t())
    cos.clamp(-1, 1)

    cos1 = cos.detach()
    cos1.scatter_(1, torch.arange(cls_num).view(-1, 1).long().cuda(), -100)

    _, indices = torch.max(cos1, dim=0)
    mask = torch.zeros((cls_num, cls_num)).cuda()
    mask.scatter_(1, indices.view(-1, 1).long(), 1)

    exclusive_loss = torch.dot(cos.view(cos.numel()), mask.view(mask.numel())) / cls_num
    return exclusive_loss
