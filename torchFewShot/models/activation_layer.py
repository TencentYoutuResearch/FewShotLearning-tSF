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

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction


def get_activation_layer(activation):
    # Create activation layer from string/function.

    # Parameters:
    # ----------
    # activation : function, or str, or nn.Module
    #     Activation function or name of activation function.

    # Returns:
    # -------
    # nn.Module
    #     Activation layer.

    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
            #return nn.ReLU()
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation


def activation_layer():
    return get_activation_layer("relu")
    #return get_activation_layer("relu6")
    #return get_activation_layer("leaky_relu")

