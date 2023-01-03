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
import numpy as np
import torch
import pickle
import cv2


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class tieredImageNet_load(object):
    # change Path
    tiered_dataset_dir = 'Path/tiered-imagenet/'

    def __init__(self, **kwargs):
        super(tieredImageNet_load, self).__init__()
        # load pkl
        self.tiered_train_dir = os.path.join(self.tiered_dataset_dir, 'train_images_png.pkl')
        self.tiered_train_label_dir = os.path.join(self.tiered_dataset_dir, 'train_labels.pkl')
        self.tiered_val_dir = os.path.join(self.tiered_dataset_dir, 'val_images_png.pkl')
        self.tiered_val_label_dir = os.path.join(self.tiered_dataset_dir, 'val_labels.pkl')
        self.tiered_test_dir = os.path.join(self.tiered_dataset_dir, 'test_images_png.pkl')
        self.tiered_test_label_dir = os.path.join(self.tiered_dataset_dir, 'test_labels.pkl')

        # train, val, test for tiered dataset
        self.tiered_train, self.tiered_train_labels2inds, self.tiered_train_labelIds = self._process_dir(
                                                        self.tiered_train_dir, self.tiered_train_label_dir)
        self.tiered_val, self.tiered_val_labels2inds, self.tiered_val_labelIds = self._process_dir(
                                                        self.tiered_val_dir, self.tiered_val_label_dir)
        self.tiered_test, self.tiered_test_labels2inds, self.tiered_test_labelIds = self._process_dir(
                                                        self.tiered_test_dir, self.tiered_test_label_dir)

        self.num_train_cats = len(self.tiered_train_labelIds)
        num_total_cats = len(self.tiered_train_labelIds) + \
                            len(self.tiered_val_labelIds) + \
                            len(self.tiered_test_labelIds)
        num_total_imgs = len(self.tiered_train + self.tiered_val + self.tiered_test)

        # TieredImageNet statistics
        print("=> TieredImageNet loaded")
        print("tiered_dataset statistics:")
        print("  ------------------------------  ")
        print("  subset   |  # cats | # images  ")
        print("  ------------------------------  ")
        print("  train    |  {:5d} | {:8d} ".format(len(self.tiered_train_labelIds), len(self.tiered_train)))
        print("  val      |  {:5d} | {:8d} ".format(len(self.tiered_val_labelIds), len(self.tiered_val)))
        print("  test     |  {:5d} | {:8d} ".format(len(self.tiered_test_labelIds), len(self.tiered_test)))
        print("  ------------------------------  ")
        print("  total    |  {:5d} | {:8d} ".format(num_total_cats, num_total_imgs))
        print("  ------------------------------  ")

    def _check_before_run(self):
        # Check if all files are available before going deeper
        if not osp.exists(self.tiered_dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.tiered_dataset_dir))
        if not osp.exists(self.tiered_train_dir):
            raise RuntimeError("'{}' is not available".format(self.tiered_train_dir))
        if not osp.exists(self.tiered_val_dir):
            raise RuntimeError("'{}' is not available".format(self.tiered_val_dir))
        if not osp.exists(self.tiered_test_dir):
            raise RuntimeError("'{}' is not available".format(self.tiered_test_dir))

    def _get_pair(self, tiered_data, labels):
        assert (tiered_data.shape[0] == len(labels))
        tiered_data_pair = []
        for i in range(tiered_data.shape[0]):
            tiered_data_pair.append((tiered_data[i], labels[i]))
        return tiered_data_pair

    def _process_dir(self, file_path, label_path):
        tiered_dataset = load_data(file_path)
        images = np.zeros([len(tiered_dataset), 84, 84, 3], dtype=np.uint8)
        for ii, item in enumerate(tiered_dataset):
            im = cv2.imdecode(item, 1)
            images[ii] = im
        print(images.shape)
        #print(images[0].shape)
        tiered_dataset_label = load_data(label_path)
        labels = np.array(tiered_dataset_label['label_specific'])
        #print(labels.shape)
        data_pair = self._get_pair(images, labels)
        labels2inds = buildLabelIndex(labels)
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds

    def _process_dir2(self, file_path1, label_path1, file_path2, label_path2):
        tiered_dataset1 = load_data(file_path1)
        images1 = np.zeros([len(tiered_dataset1), 84, 84, 3], dtype=np.uint8)
        for ii, item in enumerate(tiered_dataset1):
            im = cv2.imdecode(item, 1)
            images1[ii] = im
        tiered_dataset_label1 = load_data(label_path1)
        labels1 = np.array(tiered_dataset_label1['label_specific'])

        tiered_dataset2 = load_data(file_path2)
        images2 = np.zeros([len(tiered_dataset2), 84, 84, 3], dtype=np.uint8)
        for ii, item in enumerate(tiered_dataset2):
            im = cv2.imdecode(item, 1)
            images2[ii] = im
        tiered_dataset_label2 = load_data(label_path2)
        labels2 = np.array(tiered_dataset_label2['label_specific']) + 351
        images = np.concatenate((images1, images2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        print(images.shape)
        data_pair = self._get_pair(images, labels)
        labels2inds = buildLabelIndex(labels)
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds


if __name__ == '__main__':
    tieredImageNet_load()

