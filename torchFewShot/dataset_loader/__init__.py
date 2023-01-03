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

# init dataset_loader
from __future__ import absolute_import
from .train_loader import FewShotDataset_train
from .test_loader import FewShotDataset_test


# loader_factory for datasets
__loader_factory = {
        'train_loader': FewShotDataset_train,
        'test_loader': FewShotDataset_test,
}


# get dataset names
def get_names():
    return list(__loader_factory.keys())


# init data loader
def init_loader(name, *args, **kwargs):
    if name not in list(__loader_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __loader_factory[name](*args, **kwargs)

