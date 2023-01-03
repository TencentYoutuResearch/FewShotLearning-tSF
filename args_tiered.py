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

import time
import argparse
import torchFewShot


def argument_parser():
    parser_tiered = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # Datasets (general)
    parser_tiered.add_argument('-d', '--dataset', type=str, default='tieredImageNet_load')
    parser_tiered.add_argument('--load', default=True)

    parser_tiered.add_argument('-j', '--workers', default=8, type=int,
                        help="number of data loading workers (default: 4)")
    parser_tiered.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser_tiered.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")

    # Optimization options
    parser_tiered.add_argument('--optim', type=str, default='sgd',
                        help="optimization algorithm (see optimizers.py)")
    parser_tiered.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        help="initial learning rate")
    parser_tiered.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")

    parser_tiered.add_argument('--max-epoch', default=100, type=int,
                        help="maximum epochs to run, default=100")
    parser_tiered.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser_tiered.add_argument('--stepsize', default=[80], nargs='+', type=int,
                        help="stepsize to decay learning rate, default=[80]")
    parser_tiered.add_argument('--LUT_lr', default=[(40, 0.05), (60, 0.01), (80, 0.001), (100, 0.0001)],
                        help="multistep to decay learning rate")

    parser_tiered.add_argument('--train-batch', default=8//8, type=int,
                        help="train batch size")
    parser_tiered.add_argument('--test-batch', default=4, type=int,
                        help="test batch size")

    # Architecture settings
    parser_tiered.add_argument('--backbone', default='resnet12_gcn_640',
                        help="conv4_512, resnet12_gcn_640, wrn28_10")
    # neck
    parser_tiered.add_argument('--neck', default='tSF', help="None, tSF")
    parser_tiered.add_argument('--num_queries', default={'coarse': 5}, help="num_queries for tSF in neck")
    parser_tiered.add_argument('--num_heads', default=4, help="num_heads for tSF in neck")
    # auxiliary loss
    parser_tiered.add_argument('--using_focal_loss', default=True, help="focal loss")
    parser_tiered.add_argument('--rotation_loss', default=True, help="rotation self-supervied learning")
    parser_tiered.add_argument('--global_weighted_loss', default=True, help="global automatic weighted loss")
    # other setting
    parser_tiered.add_argument('--num_classes', type=int, default=351, help="num_classes: train=351, val=97")
    parser_tiered.add_argument('--scale_cls', type=int, default=7)

    # Miscs
    parser_tiered.add_argument('--save-dir', type=str, default='save-dir')
    parser_tiered.add_argument('--fix_backbone', default=False, help="fix backbone in training")
    parser_tiered.add_argument('--load_backbone', default=False, help="only load the backbone in training")
    parser_tiered.add_argument('--resume', type=str, default='None', metavar='PATH')
    parser_tiered.add_argument('--gpu-devices', default=[0,1,2,3,4,5,6,7])
    parser_tiered.add_argument('--norm_layer', type=str, default='torchsyncbn', help="bn, in, torchsyncbn")
    parser_tiered.add_argument('--local_rank')

    # FewShot settting
    parser_tiered.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser_tiered.add_argument('--nExemplars', type=int, default=1,
                        help='number of training examples per novel category.')

    parser_tiered.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                        help='number of test examples for all the novel category when training')
    parser_tiered.add_argument('--train_epoch_size', type=int, default=13980,
                        help='number of batches per epoch when training')
    parser_tiered.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser_tiered.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch')

    parser_tiered.add_argument('--phase', default='test', type=str,
                        help='use test or val dataset to early stop') # default='test' of CAN origin
    parser_tiered.add_argument('--seed', type=int, default=1)

    return parser_tiered

