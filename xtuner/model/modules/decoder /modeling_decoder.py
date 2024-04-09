import torch
import torch.nn.functional as F
from torch import Tensor, nn
from zeta.nn import FeedForward, MultiQueryAttention

"""
Ref:
    https://github.com/kyegomez/MoE-Mamba/blob/bba93e4b0b797c0bd2d509f8b844e7845b6127d4/moe_mamba/model.py#L79
    https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py

"""
