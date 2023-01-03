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

from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random
import cv2

import torch
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .train import test
sys.path.append('./torchFewShot')
from torchFewShot.models.net import Model
from torchFewShot.data_manager import DataManager
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate
from torchFewShot.utils.mkdir import check_mkdir, check_makedirs
from tensorboardX import SummaryWriter

# from test_tiered_args import argument_parser
from test_mini_args import argument_parser


parser = argument_parser()
args_test = parser.parse_args()


def main_test():
    if args_test.norm_layer != 'torchsyncbn':
        torch.manual_seed(args_test.seed)
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args_test.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args_test))

    print("Currently using GPU {}".format(args_test.gpu_devices))
    cudnn.benchmark = True
    if args_test.norm_layer != 'torchsyncbn':
        torch.cuda.manual_seed_all(args_test.seed)
    
    device = None
    if args_test.norm_layer == 'torchsyncbn':
        local_rank = int(args_test.local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        print(f"[init distributed device] == local rank: {local_rank} ==")

    print('Initializing image data manager')
    dm_test = DataManager(args_test, use_gpu)
    _, testloader = dm_test.return_dataloaders()

    # define model_test
    model_test = Model(args=args_test)
    # DataParallel
    if len(args_test.gpu_devices) > 1:
        print("=> {} GPU parallel".format(len(args_test.gpu_devices)))
        if args_test.norm_layer == 'bn':
            model_test = nn.DataParallel(model_test, device_ids=args_test.gpu_devices)
        elif args_test.norm_layer == 'torchsyncbn':
            # DistributedDataParallel
            model_test = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_test)
            model_test = model_test.to(device)
            model_test = DDP(model_test, device_ids=[device], output_device=device, find_unused_parameters=True)
        # load the model_test
        checkpoint = torch.load(args_test.resume, map_location=device)
    else:
        # load the model_test
        checkpoint = torch.load(args_test.resume, map_location='cuda:0')
    model_test.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Loaded checkpoint from '{}'".format(args_test.resume))

    # test
    model_test = model_test.cuda()
    test(args_test, model_test, testloader, use_gpu, device)


if __name__ == '__main__':
    main_test()
