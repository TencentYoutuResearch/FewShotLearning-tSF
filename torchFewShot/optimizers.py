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
import torch


def init_optimizer(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                lr=lr, weight_decay=weight_decay)
    elif optim == 'amsgrad':
        return torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim == 'sgd':
        return torch.optim.SGD(filter(lambda p: p.requires_grad, params),
                    lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(filter(lambda p: p.requires_grad, params),
                    lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optimizer: {}".format(optim))

