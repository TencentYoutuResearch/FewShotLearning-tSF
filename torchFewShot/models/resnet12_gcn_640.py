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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation_layer import activation_layer


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# BasicBlock of ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel=3, stride=1,
                downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        if kernel == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        if kernel == 1:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv3 = conv3x3(planes, planes)
        self.bn3 = self.norm_layer(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = activation_layer()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, kernel=3, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        self.kernel = kernel
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = activation_layer()

        # make layer
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 160, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 320, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 640, layers[3], stride=2)

        self.nFeat = 640 * block.expansion # output feature channel

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.norm_layer):
                if not isinstance(m, nn.InstanceNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.AvgPool2d(kernel_size=stride, stride=stride,
                #     ceil_mode=True, count_include_pad=False),
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #     kernel_size=1, stride=1, bias=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.kernel, stride,
                        downsample, norm_layer = self.norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.kernel, norm_layer = self.norm_layer))

        return nn.Sequential(*layers)

    def f_shuffle(self, f):
        B, c, h, w = f.size()
        f = f.contiguous().view(B, c, h*w)
        groups = h
        f = f.view(B,c,groups,h*w//groups).permute(0,1,3,2).contiguous().view(B, c, h, w)
        return f

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4


def resnet12_gcn_640(norm_layer=nn.BatchNorm2d):
    model = ResNet(BasicBlock, [1,1,1,1], kernel=3, norm_layer=norm_layer) # resnet12
    #model = ResNet(BasicBlock, [2,2,2,2], kernel=3, norm_layer=norm_layer) # resnet18
    return model

