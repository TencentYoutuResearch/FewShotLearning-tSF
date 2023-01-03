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
from torch.autograd import Variable
import numpy as np

from .conv4_512 import conv4_512
from .resnet12_gcn_640 import resnet12_gcn_640
from .wrn28_10 import wrn28_10
from .neck import tSF


class SimilarityResFunc(nn.Module):
    def __init__(self, args, backbone_outsize, scale_cls, norm_layer=nn.BatchNorm2d):
        super(SimilarityResFunc, self).__init__()
        self.backbone_outsize = backbone_outsize
        self.scale_cls = scale_cls
        self.norm_layer = norm_layer
        self.num_classes = args.num_classes

        # global loss
        self.clasifier = nn.Conv2d(self.backbone_outsize[0], self.num_classes, kernel_size=1)
        # rotation loss
        self.rotation_loss = args.rotation_loss
        if self.rotation_loss:
            self.rotate_classifier = nn.Conv2d(self.backbone_outsize[0], 4, kernel_size=1)


    def test(self, ftrain, ftest):
        ftrain = ftrain.view(*ftrain.size()[:4], -1).mean(4)
        ftest = ftest.view(*ftest.size()[:4], -1).mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, ytest, ftrain, ftest, pids=None):
        # output results init####
        output_results = {
            'ytest': None,
            'cls_scores': None,
            'rotate_scores': None,
            'metric_scores': None
        }

        # reshape
        b, n1, c, h, w = ftrain.size()
        n2 = ftest.size(1)
        ftrain = ftrain.view(b, n1, c, -1)
        ftest = ftest.view(b, n2, c, -1)
        ftrain = ftrain.unsqueeze(2).repeat(1,1,n2,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
        ftest = ftest.unsqueeze(1).repeat(1,n1,1,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
        b, n2, n1, c, h, w = ftrain.size()
        # testing
        if not self.training:
            global_scores = self.test(ftrain, ftest) # torch.Size([4, 75, 5])
            output_results['metric_scores'] = global_scores
            return output_results
        # training
        # loss
        # cos similarity (patch loss)
        ftrain = ftrain.view(*ftrain.size()[:4], -1).mean(4)
        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(b * n2, *cls_scores.size()[2:])
        # similarity losses
        output_results['cls_scores'] = cls_scores
        # output results loss
        # global loss
        ftest = ftest.contiguous().view(b, n2, n1, -1)
        ftest = ftest.transpose(2, 3)
        ytest = ytest.unsqueeze(3)
        ftest = torch.matmul(ftest, ytest)
        ftest = ftest.view(b * n2, -1, h, w)
        ytest = self.clasifier(ftest)
        if self.rotation_loss:
            # rotation loss
            rotate_scores = self.rotate_classifier(ftest)

        # output results loss
        # global losses
        output_results['ytest'] = ytest
        # other losses
        if self.rotation_loss:
            output_results['rotate_scores'] = rotate_scores
        return output_results


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.backbone = args.backbone
        self.neck = args.neck
        self.num_queries = args.num_queries
        self.num_heads = args.num_heads
        self.scale_cls = args.scale_cls
        self.fix_backbone = args.fix_backbone
        if args.norm_layer == 'bn':
            self.norm_layer = nn.BatchNorm2d
        elif args.norm_layer == 'in':
            self.norm_layer = nn.InstanceNorm2d
        elif args.norm_layer == 'torchsyncbn':
            self.norm_layer = nn.BatchNorm2d
        # backbone
        if self.backbone == 'conv4_512':
            self.base = conv4_512()
            self.backbone_outsize = [512,19,19]
        elif self.backbone == 'resnet12_gcn_640':
            self.base = resnet12_gcn_640(norm_layer=self.norm_layer) # self.base.nFeat = 640
            self.backbone_outsize = [self.base.nFeat,11,11] # layer1 stride=1
        elif self.backbone == 'wrn28_10':
            self.base = wrn28_10(norm_layer=self.norm_layer) # self.base.nFeat = 640
            self.backbone_outsize = [self.base.nFeat,7,7]

        # neck
        if self.neck == 'tSF':
            self.neck_tSF = tSF(feature_dim=self.backbone_outsize[0],
                                num_queries=self.num_queries['coarse'],
                                num_heads=self.num_heads, FFN_method='MLP')

        # fix the layers above (including backbone) in training
        if self.fix_backbone:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True

        # similairty results function
        self.similarity_res_func = SimilarityResFunc(args, self.backbone_outsize,
                                    self.scale_cls, norm_layer=self.norm_layer)


    def f_to_train_test(self, f, ytrain, batch_size, num_train, num_test):
        # class mean
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.contiguous().view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.contiguous().view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.contiguous().view(batch_size, num_test, *f.size()[1:])
        return ftrain, ftest

    def forward(self, xtrain, xtest, ytrain, ytest, pids=None):
        # feature
        b, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        B, c, h, w = f.size()

        # neck: tSF
        if self.neck == 'tSF':
            f, softmax_qk_coarse = self.neck_tSF(f)

        # transfer f to ftrain and ftest
        ftrain, ftest = self.f_to_train_test(f, ytrain, b, num_train, num_test)

        # testing and loss
        output_results = self.similarity_res_func(ytest, ftrain, ftest, pids=pids)
        return output_results


if __name__ == '__main__':
    torch.manual_seed(0)
    net = Model()
    net.eval()

