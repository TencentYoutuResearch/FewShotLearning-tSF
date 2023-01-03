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
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
#import lmdb
import io
import random

import torch
from torch.utils.data import Dataset
from train_loader import FewShotDataset_train


# read_image
def read_image(img_path):
    # Keep reading image until succeed.
    # This can avoid IOError incurred by heavy IO process.
    got_img = False
    if not osp.exists(img_path):
        # IOError
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            # IOError
            print("IOError incurred when reading '{}'. Will redo. Don't worry.".format(img_path))
            pass
    return img


# FewShotDataset_test
class FewShotDataset_test(FewShotDataset_train):
    # Few shot epoish Dataset

    # Returns a task (Xtrain, Ytrain, Xtest, Ytest) to classify'
    #     Xtrain: [nKnovel*nExpemplars, c, h, w].
    #     Ytrain: [nKnovel*nExpemplars].
    #     Xtest:  [nTestNovel, c, h, w].
    #     Ytest:  [nTestNovel].

    def __init__(self,
                 dataset, # dataset of [(img_path, cats), ...].
                 labels2inds, # labels of index {(cats: index1, index2, ...)}.
                 labelIds, # train labels [0, 1, 2, 3, ...,].
                 nKnovel=5, # number of novel categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=2*5, # number of test examples for all the novel categories.
                 epoch_size=2000, # number of tasks per eooch.
                 transform=None,
                 load=True,
                 **kwargs
                 ):
        
        self.dataset = dataset
        self.labels2inds = labels2inds
        self.labelIds = labelIds
        self.nKnovel = nKnovel
        self.transform = transform

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size
        self.load = load

        seed = 112
        random.seed(seed)
        np.random.seed(seed)

        self.Epoch_Exemplar = []
        self.Epoch_Tnovel = []
        for i in range(epoch_size):
            Tnovel, Exemplar = self._sample_episode()
            self.Epoch_Exemplar.append(Exemplar)
            self.Epoch_Tnovel.append(Tnovel)

    def _creatExamplesTensorData(self, examples):
        # Creats the examples image label tensor data.

        # Args:
        #     examples: a list of 2-element tuples. (sample_index, label).

        # Returns:
        #     images: a tensor [nExemplars, c, h, w]
        #     labels: a tensor [nExemplars]

        images = []
        labels = []
        for (img_idx, label) in examples:
            img = self.dataset[img_idx][0]
            if self.load:
                img = Image.fromarray(img)
            else:
                img = read_image(img)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        return images, labels

    def __getitem__(self, index):
        Tnovel = self.Epoch_Tnovel[index]
        Exemplars = self.Epoch_Exemplar[index]
        Xt, Yt = self._creatExamplesTensorData(Exemplars)
        Xe, Ye = self._creatExamplesTensorData(Tnovel)
        return Xt, Yt, Xe, Ye

