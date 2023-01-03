# Copyright. All rights reserved.
# If you use this code for your research, please cite our paper:
# @inproceedings{jinxiang2022tSF,
#   title={tSF: Transformer-based Semantic Filter for Few-Shot Learning},
#   author={Jinxiang, Lai and Siqian, Yang and Wenlong, Liu and # NOCA:InnerUsernameLeak(论文署名)
#             Yi, Zeng and Zhongyi, Huang and Wenlong, Wu and # NOCA:InnerUsernameLeak(论文署名)
#             Jun, Liu and Bin-Bin, Gao and Chengjie, Wang}, # NOCA:InnerUsernameLeak(论文署名)
#   booktitle={ECCV},
#   year={2022}
# }

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random

import math
from torch.nn.utils.weight_norm import WeightNorm


act = torch.nn.ReLU()


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        self.bn1 = self.norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes,
                            kernel_size=1, stride=stride,
                            padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0,
                norm_layer=nn.BatchNorm2d):
        super(NetworkBlock, self).__init__()
        self.norm_layer = norm_layer
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and
                                stride or 1, dropRate, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, norm_layer=nn.BatchNorm2d, stride=1):
        dropRate = 0.5
        flatten = True
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.norm_layer = norm_layer
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride,
                                    dropRate, norm_layer=self.norm_layer)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                    dropRate, norm_layer=self.norm_layer)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
                                    dropRate, norm_layer=self.norm_layer)
        # global average pooling and linear
        self.bn1 = self.norm_layer(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.nFeat = nChannels[3]
        if flatten:
            self.final_feat_dim = 640
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.norm_layer):
                if not isinstance(m, nn.InstanceNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out)) # (21,21)
        out = F.avg_pool2d(out,kernel_size=3) # (7,7)
        return out


def wrn28_10(norm_layer=nn.BatchNorm2d):
    model = WideResNet(depth=28, widen_factor=10, norm_layer=norm_layer, stride=1)
    return model

