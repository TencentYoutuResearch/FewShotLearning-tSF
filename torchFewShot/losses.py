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

from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, using_focal_loss=False):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.using_focal_loss = using_focal_loss

    def forward(self, inputs, targets, teacher_preds=None, teacher_targets=None):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        margin = 0 # default = 0
        log_probs = self.logsoftmax(inputs) + margin
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        if teacher_preds != None and teacher_targets != None:
            teacher_preds = teacher_preds.view(teacher_preds.size(0), teacher_preds.size(1), -1)
            teacher_preds = self.softmax(teacher_preds)

            teacher_targets = torch.zeros(teacher_preds.size(0), teacher_preds.size(1)).scatter_(
                                            1, teacher_targets.unsqueeze(1).data.cpu(), 1)
            teacher_targets = teacher_targets.unsqueeze(-1)
            teacher_targets = teacher_targets.cuda()
            teacher_targets_org = teacher_targets.repeat(1,1,teacher_preds.size(2))
            teacher_targets = teacher_targets * teacher_preds

            # label mapping
            targets_org = targets.repeat(1,1,inputs.size(2))
            targets = targets * inputs
            targets[targets_org == 1] = teacher_targets[teacher_targets_org == 1]
            # threshold
            # targets[targets>0.5] = 1
            thres_scale = 100.0
            targets[targets * thres_scale > 1] = 1
            targets[targets * thres_scale < 1] = 0
        # focal loss
        # using_focal_loss = False
        if self.using_focal_loss:
            gama = 1 # default = 2
            probs = F.softmax(inputs, dim=1)
            loss = (- targets * torch.pow((1-probs), gama) * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).mean(0).sum()
        return loss / inputs.size(2)


# AutomaticMetricLoss
class AutomaticMetricLoss(nn.Module):
    def __init__(self, num=2, init_weight=1.0, min_weights=[0,0]):
        super(AutomaticMetricLoss, self).__init__()
        self.num = num
        self.min_weights = min_weights
        params = torch.ones(num, requires_grad=True) * init_weight
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        weights = []
        bias = []
        for i, loss in enumerate(x):
            weights_i = 0.5 / (self.params[i] ** 2) + self.min_weights[i]
            bias_i = torch.log(1 + self.params[i] ** 2)
            loss_sum += weights_i * loss + bias_i
            weights.append(weights_i)
            bias.append(bias_i)
        return loss_sum, weights, bias

