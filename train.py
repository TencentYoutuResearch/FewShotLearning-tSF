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

import torch
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append('./torchFewShot')

# from args_tiered import argument_parser
# from args_tiered_no_warmup import argument_parser
from args_mini import argument_parser
# from args_mini_fine import argument_parser
# from args_cifar import argument_parser

from torchFewShot.models.net import Model
from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss, AutomaticMetricLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate


parser = argument_parser()
args = parser.parse_args()


def main_train():
    if args.norm_layer != 'torchsyncbn':
        torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    print("Currently using GPU {}".format(args.gpu_devices))
    cudnn.benchmark = True
    if args.norm_layer != 'torchsyncbn':
        torch.cuda.manual_seed_all(args.seed)

    device = None
    if args.norm_layer == 'torchsyncbn':
        # 0. set up distributed device
        # rank = int(os.environ["RANK"])
        # local_rank = int(os.environ["LOCAL_RANK"])
        local_rank = int(args.local_rank)
        # torch.cuda.set_device(rank % torch.cuda.device_count())
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        print(f"[init distributed device] == local rank: {local_rank} ==")
        # print(f"GPU num: {torch.cuda.device_count()}")

    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    # define model
    model = Model(args=args)
    # DataParallel
    if len(args.gpu_devices) > 1:
        print("=> {} GPU parallel".format(len(args.gpu_devices)))
        if args.norm_layer == 'bn':
            model = nn.DataParallel(model, device_ids=args.gpu_devices)
        elif args.norm_layer == 'torchsyncbn':
            # DistributedDataParallel
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(device)
            # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    criterion = CrossEntropyLoss(args.using_focal_loss)
    awl_global = None
    if args.global_weighted_loss:
        metric_weight = 0.0
        auxiliary_weight = 0.5 # for ResNet-12
        # auxiliary_weight = 1.5 # for WRN-28
        awl_global = AutomaticMetricLoss(num=2, init_weight=1.0, min_weights=[auxiliary_weight,auxiliary_weight])

    if not args.global_weighted_loss:
        optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr,
            'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True},
            {'params': awl_global.parameters(), 'lr': args.lr,
            'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True}])

    model = model.cuda()
    if args.global_weighted_loss:
        awl_global = awl_global.cuda()

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)
        if args.norm_layer == 'torchsyncbn':
            # set sampler
            trainloader.sampler.set_epoch(epoch)

        start_train_time = time.time()
        train(args, epoch, model, criterion, awl_global, optimizer, trainloader, learning_rate, use_gpu, device)
        train_time += round(time.time() - start_train_time)

        if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
            acc = test(args, model, testloader, use_gpu, device)
            is_best = acc > best_acc

            if is_best:
                best_acc = acc
                best_epoch = epoch + 1

                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'best_model.pth.tar'))
                # is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar')

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(args, epoch, model, criterion, awl_global, optimizer, trainloader, learning_rate, use_gpu, device):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids) in enumerate(trainloader):
        data_time.update(time.time() - end)
        batch_size, num_train_examples, channels, height, width = images_train.size()
        test_batch_size, num_test_examples = images_test.size(0), images_test.size(1)
        if args.rotation_loss:
            total_test_num = test_batch_size * num_test_examples
            x, y = images_test.view(total_test_num, channels, height, width), labels_test.view(total_test_num)
            y_pids = pids.view(total_test_num)
            x_, y_, y_pids_, a_ = [], [], [], []
            for j in range(total_test_num):
                x90 = x[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                x_ += [x[j], x90, x180, x270]
                y_ += [y[j] for _ in range(4)]
                y_pids_ += [y_pids[j] for _ in range(4)]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]
            x_ = Variable(torch.stack(x_,0)).view(test_batch_size, num_test_examples*4, channels, height, width)
            y_ = Variable(torch.stack(y_,0)).view(test_batch_size, num_test_examples*4)
            y_pids_ = Variable(torch.stack(y_pids_,0)).view(test_batch_size, num_test_examples*4)
            a_ = Variable(torch.stack(a_,0)).view(test_batch_size, num_test_examples*4)
            images_test, labels_test, pids = x_, y_, y_pids_
            if use_gpu:
                a_ = a_.cuda()
        if use_gpu:
            if args.norm_layer == 'torchsyncbn':
                images_train, labels_train = images_train.to(device), labels_train.to(device)
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                labels_train_1hot = one_hot(labels_train).to(device)
                labels_test_1hot = one_hot(labels_test).to(device)
                pids = pids.to(device)
            else:
                images_train, labels_train = images_train.cuda(), labels_train.cuda()
                images_test, labels_test = images_test.cuda(), labels_test.cuda()
                labels_train_1hot = one_hot(labels_train).cuda()
                labels_test_1hot = one_hot(labels_test).cuda()
                pids = pids.cuda()
        # model
        output_results = model(images_train, images_test, labels_train_1hot, labels_test_1hot, pids=pids)
        # losses
        if args.rotation_loss:
            global_loss_scale = 1.0 # default=1.0
            metric_loss_scale = 0.5 # default=0.5
            rotate_loss_scale = 1.0 # default=1.0
        else:
            global_loss_scale = 1.0 # default=1.0
            metric_loss_scale = 0.5 # default=0.5
            rotate_loss_scale = 0.0 # default=0.0
            args.global_weighted_loss = False
        loss1 = criterion(output_results['ytest'], pids.view(-1))
        loss2 = criterion(output_results['cls_scores'], labels_test.view(-1))
        if not args.global_weighted_loss:
            loss = global_loss_scale * loss1 + metric_loss_scale * loss2
        if args.rotation_loss: # rotate loss
            loss_rotate = criterion(output_results['rotate_scores'], a_.view(-1))
            if not args.global_weighted_loss:
                loss = loss + rotate_loss_scale * loss_rotate # default=1.0
        if args.rotation_loss and args.global_weighted_loss: # global weighted losses
            loss, loss_weights_tmp, loss_bias_tmp = awl_global(loss1, loss_rotate)
            loss = loss + metric_loss_scale * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
           epoch+1, learning_rate, batch_time=batch_time, data_time=data_time, loss=losses))


def test(args, model, testloader, use_gpu, device):
    accs = AverageMeter()
    test_accuracies = []

    model_dir = os.path.dirname(args.resume)
    for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(testloader):
        if use_gpu:
            if args.norm_layer == 'torchsyncbn':
                images_train = images_train.to(device)
                images_test = images_test.to(device)
            else:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

        end = time.time()

        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)
        labels_train_org = labels_train

        if args.norm_layer == 'torchsyncbn':
            labels_train_1hot = one_hot(labels_train).to(device)
            labels_test_1hot = one_hot(labels_test).to(device)
        else:
            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

        # testing
        model.eval()
        with torch.no_grad():
            output_results = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            cls_scores = output_results['metric_scores']
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)

            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            preds_org = preds
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))
            # print('accs.avg: {:.2%}'.format(accs.avg))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() # [b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

    # result
    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))

    return accuracy


if __name__ == '__main__':
    main_train()

