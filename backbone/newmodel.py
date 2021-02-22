import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Parameter
import bagnets.pytorchnet


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FiberNet(nn.Module):
    def __init__(self, patch_size, feature_dim=512, drop_ratio=0.4):
        super(FiberNet, self).__init__()
        assert patch_size in [9, 17, 33]
        if patch_size == 9:
            bagnet = bagnets.pytorchnet.bagnet9(pretrained=False)
            self.weight = nn.Parameter(torch.Tensor(1, 1, 13, 13), requires_grad=True)
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(drop_ratio),
                                              Flatten(),
                                              nn.Linear(512 * 13 * 13, feature_dim),
                                              nn.BatchNorm1d(feature_dim))
        elif patch_size == 17:
            bagnet = bagnets.pytorchnet.bagnet17(pretrained=False)
            self.weight = nn.Parameter(torch.Tensor(1, 1, 12, 12), requires_grad=True)
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(drop_ratio),
                                              Flatten(),
                                              nn.Linear(512 * 12 * 12, feature_dim),
                                              nn.BatchNorm1d(feature_dim))
        elif patch_size == 33:
            bagnet = bagnets.pytorchnet.bagnet33(pretrained=False)
            self.weight = nn.Parameter(torch.Tensor(1, 1, 12, 12), requires_grad=True)
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(drop_ratio),
                                              Flatten(),
                                              nn.Linear(512 * 12 * 12, feature_dim),
                                              nn.BatchNorm1d(feature_dim))
        else:
            print("patch_size : 9, 17, 33")
        nn.init.xavier_uniform_(self.weight)

        self.relu = nn.ReLU()
        self.bagnet_backbone = nn.Sequential(*list(bagnet.children())[0:-3])
        self.bagnet_bn_4 = nn.BatchNorm2d(1024)
        self.bagnet_last = nn.Conv2d(1024, 512, 1)

    def forward(self, x):
        x = self.bagnet_backbone(x)
        x = self.bagnet_bn_4(x)
        x = self.bagnet_last(x)

        activated_weight = self.relu(self.weight)
        x = torch.mul(x, activated_weight)
        x = self.output_layer(x)
        return x
