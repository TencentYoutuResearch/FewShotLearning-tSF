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

# init datasets
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .miniImageNet_load import miniImageNet_load
from .tieredImageNet_load import tieredImageNet_load


# imgfewshot_factory
__imgfewshot_factory = {
        'miniImageNet_load': miniImageNet_load,
        'tieredImageNet_load': tieredImageNet_load,
}


# get dataset names
def get_names():
    return list(__imgfewshot_factory.keys())


# init_imgfewshot_dataset
def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(
                        name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](**kwargs)

