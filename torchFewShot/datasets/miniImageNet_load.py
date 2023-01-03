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
from __future__ import print_function

import os
import torch
import pickle
import numpy as np


# load_data
def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='iso-8859-1')
    return data


# buildLabelIndex
def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


# miniImageNet_load
class miniImageNet_load(object):
    # Dataset statistics:
    # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)

    # change Path
    dataset_dir = 'Path/MiniImagenet/'

    def __init__(self, **kwargs):
        super(miniImageNet_load, self).__init__()
        # load pickle
        self.train_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_train_phase_train.pickle')
        self.val_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_val.pickle')
        self.test_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_test.pickle')

        # get train, val, test
        self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir)
        self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir)
        self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir)

        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        # MiniImageNet statistics
        print("=> MiniImageNet loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds), len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds), len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def _check_before_run(self):
        # Check if all files are available before going deeper
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _get_pair(self, data, labels):
        assert (data.shape[0] == len(labels))
        data_pair = []
        for i in range(data.shape[0]):
            data_pair.append((data[i], labels[i]))
        return data_pair

    def _process_dir(self, file_path):
        dataset = load_data(file_path)
        data = dataset['data']
        print(data.shape)
        labels = dataset['labels']
        data_pair = self._get_pair(data, labels)
        labels2inds = buildLabelIndex(labels)
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds

    def _process_dir2(self, file_path1, file_path2):
        dataset1 = load_data(file_path1)
        data1 = dataset1['data']
        labels1 = dataset1['labels']
        dataset2 = load_data(file_path2)
        data2 = dataset2['data']
        labels2 = dataset2['labels']
        data = np.concatenate((data1, data2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        print(data.shape)
        data_pair = self._get_pair(data, labels)
        labels2inds = buildLabelIndex(labels)
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds


if __name__ == '__main__':
    # example
    miniImageNet_load()

