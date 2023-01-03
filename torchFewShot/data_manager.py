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
from __future__ import print_function

import torch
from torch.utils.data import DataLoader

import transforms as T
import datasets
import dataset_loader


# DataManager
class DataManager(object):
    # Few shot data manager
    def __init__(self, args, use_gpu):
        super(DataManager, self).__init__()
        self.args = args
        self.use_gpu = use_gpu
        # Initializing dataset
        print("Initializing dataset {}".format(args.dataset))
        dataset = datasets.init_imgfewshot_dataset(name=args.dataset)
        transform_train = T.Compose([T.RandomCrop(args.height, padding=8),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.RandomHorizontalFlip(), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(0.5)])
        transform_test = T.Compose([T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        pin_memory = True if use_gpu else False
        # train
        if args.norm_layer == 'torchsyncbn':
            # DistributedSampler
            dataset_train = dataset_loader.init_loader(name='train_loader',
                dataset=dataset.train, labels2inds=dataset.train_labels2inds,
                labelIds=dataset.train_labelIds, nKnovel=args.nKnovel,
                nExemplars=args.nExemplars, nTestNovel=args.train_nTestNovel,
                epoch_size=args.train_epoch_size, transform=transform_train, load=args.load)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train,shuffle=True)
            self.trainloader = DataLoader(dataset_train,
                batch_size=args.train_batch, shuffle=False, num_workers=args.workers,
                pin_memory=pin_memory, sampler=train_sampler, drop_last=True)
        else:
            self.trainloader = DataLoader(
                dataset_loader.init_loader(name='train_loader', dataset=dataset.train,
                    labels2inds=dataset.train_labels2inds, labelIds=dataset.train_labelIds,
                    nKnovel=args.nKnovel, nExemplars=args.nExemplars,
                    nTestNovel=args.train_nTestNovel, epoch_size=args.train_epoch_size,
                    transform=transform_train, load=args.load),
                batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=True)
        # val
        self.valloader = DataLoader(
            dataset_loader.init_loader(name='test_loader', dataset=dataset.val,
                labels2inds=dataset.val_labels2inds, labelIds=dataset.val_labelIds,
                nKnovel=args.nKnovel, nExemplars=args.nExemplars,
                nTestNovel=args.nTestNovel, epoch_size=args.epoch_size,
                transform=transform_test, load=args.load),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False)
        # test
        self.testloader = DataLoader(
            dataset_loader.init_loader(name='test_loader', dataset=dataset.test,
               labels2inds=dataset.test_labels2inds, labelIds=dataset.test_labelIds,
               nKnovel=args.nKnovel, nExemplars=args.nExemplars,
               nTestNovel=args.nTestNovel, epoch_size=args.epoch_size,
               transform=transform_test, load=args.load),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False)

    def return_dataloaders(self):
        if self.args.phase == 'test': # for test
            return self.trainloader, self.testloader
        elif self.args.phase == 'val': # for val
            return self.trainloader, self.valloader

