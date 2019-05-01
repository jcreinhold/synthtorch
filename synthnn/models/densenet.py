#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.models.densenet

holds the architecture for a 2d densenet [1]
this model is pulled (and modified) from the pytorch repo:
https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

References:
    [1] Huang, Gao, et al. "Densely connected convolutional networks."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 8, 2018
"""

__all__ = ['DenseNet']

from collections import OrderedDict
import logging
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..util.helper import get_loss

logger = logging.getLogger(__name__)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features:int, growth_rate:int, bn_size:int, drop_rate:float):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers:int, num_input_features:int, bn_size:int, growth_rate:int, drop_rate:float):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features:int, num_output_features:int):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class DenseNet(nn.Module):
    """
    Densenet-BC model class, adapted for synthesis, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int): how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints): how many layers in each pooling block
        num_init_features (int): the number of filters to learn in the first convolution layer
        bn_size (int): multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float): dropout rate after each dense layer
    """

    def __init__(self, growth_rate:int=4, block_config:Tuple[int,int,int,int]=(6, 6, 6, 6),
                 num_init_features:int=32, bn_size:int=4, dropout_prob:float=0, n_input:int=1, n_output:int=1,
                 loss:Optional[str]=None, **kwargs):

        super(DenseNet, self).__init__()
        self.criterion = get_loss(loss)

        # First convolution
        self.layers = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(n_input, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=dropout_prob)
            self.layers.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.layers.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final layer
        self.layers.add_module('final', nn.Conv2d(num_features, n_output, kernel_size=1, bias=True))


    def forward(self, x):
        out = self.layers(x)
        return out

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward(x)
