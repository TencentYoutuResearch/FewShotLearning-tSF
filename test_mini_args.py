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
    parser_mini_test = argparse.ArgumentParser(description='Test image model with 5-way classification')
    # Datasets
    parser_mini_test.add_argument('-d', '--dataset', type=str, default='miniImageNet_load')
    parser_mini_test.add_argument('--load', default=True)
    parser_mini_test.add_argument('-j', '--workers', default=8, type=int,
                        help="number of data loading workers (default: 4)")
    parser_mini_test.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser_mini_test.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")
    # Optimization options
    parser_mini_test.add_argument('--optim', type=str, default='sgd',
                        help="optimization algorithm (see optimizers.py), default=sgd")
    parser_mini_test.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        help="initial learning rate, default=0.1")
    parser_mini_test.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    parser_mini_test.add_argument('--train-batch', default=4//4, type=int,
                        help="train batch size")
    parser_mini_test.add_argument('--test-batch', default=4//4, type=int,
                        help="test batch size")
    # Architecture
    parser_mini_test.add_argument('--backbone', default='resnet12_gcn_640',
                        help="conv4_512, resnet12_gcn_640, wrn28_10")
    # neck
    parser_mini_test.add_argument('--neck', default='tSF', help="None, tSF")
    parser_mini_test.add_argument('--num_queries', default={'coarse': 5}, help="num_queries for tSF in neck")
    parser_mini_test.add_argument('--num_heads', default=4, help="num_heads for tSF in neck")
    # auxiliary loss
    parser_mini_test.add_argument('--using_focal_loss', default=True, help="focal loss")
    parser_mini_test.add_argument('--rotation_loss', default=True, help="rotation self-supervied learning")
    parser_mini_test.add_argument('--global_weighted_loss', default=True, help="global automatic weighted loss")
    # other setting
    parser_mini_test.add_argument('--num_classes', type=int, default=64)
    parser_mini_test.add_argument('--scale_cls', type=int, default=7)
    parser_mini_test.add_argument('--save-dir', type=str, default='save-dir')
    parser_mini_test.add_argument('--fix_backbone', default=True, help="fix backbone in training")
    parser_mini_test.add_argument('--load_backbone', default=False, help="only load the backbone in training")
    parser_mini_test.add_argument('--resume', type=str, default='resume/best_model.pth.tar', metavar='PATH')
    # FewShot settting
    parser_mini_test.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser_mini_test.add_argument('--nExemplars', type=int, default=5,
                        help='number of training examples per novel category.')
    parser_mini_test.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                        help='number of test examples for all the novel category when training')
    parser_mini_test.add_argument('--train_epoch_size', type=int, default=1200,
                        help='number of episodes per epoch when training')
    parser_mini_test.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser_mini_test.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch')
    # Miscs
    parser_mini_test.add_argument('--phase', default='test', type=str)
    parser_mini_test.add_argument('--seed', type=int, default=1)
    parser_mini_test.add_argument('--gpu-devices', default=[0,1])
    parser_mini_test.add_argument('--norm_layer', type=str, default='torchsyncbn', help="bn, in, torchsyncbn")
    parser_mini_test.add_argument('--local_rank')

    return parser_mini_test

