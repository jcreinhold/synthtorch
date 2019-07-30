#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.models.nconvnet

define the class for a N layer CNN with
no max pool, increase in channels, or any of that
fancy stuff. This is generally used for testing
purposes

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['SimpleConvNet']

from typing import Tuple

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SimpleConvNet(torch.nn.Module):
    def __init__(self, n_layers:int, n_input:int=1, n_output:int=1, kernel_size:Tuple[int]=(3,3,3),
                 dropout_prob:float=0, dim:int=3, **kwargs):
        super(SimpleConvNet, self).__init__()
        self.n_layers = n_layers
        self.n_input = n_input
        self.n_output = n_output
        self.kernel_sz = kernel_size
        self.dropout_prob = dropout_prob
        self.dim = dim
        self.criterion = nn.MSELoss()
        if isinstance(kernel_size[0], int):
            self.kernel_sz = [kernel_size for _ in range(n_layers)]
        else:
            self.kernel_sz = kernel_size
        pad = nn.ReplicationPad3d if dim == 3 else \
              nn.ReplicationPad2d if dim == 2 else \
              nn.ReplicationPad1d
        self.layers = nn.ModuleList([nn.Sequential(
            pad([ks//2 for p in zip(ksz,ksz) for ks in p]),
            nn.Conv3d(n_input, n_output, ksz) if dim == 3 else \
            nn.Conv2d(n_input, n_output, ksz) if dim == 2 else \
            nn.Conv1d(n_input, n_output, ksz),
            nn.ReLU(),
            nn.InstanceNorm3d(n_output, affine=True) if dim == 3 else \
            nn.InstanceNorm2d(n_output, affine=True) if dim == 2 else \
            nn.InstanceNorm1d(n_output, affine=True),
            nn.Dropout3d(dropout_prob) if dim == 3 else \
            nn.Dropout2d(dropout_prob) if dim == 2 else \
            nn.Dropout(dropout_prob)) for ksz in self.kernel_sz])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward(x)

    def freeze(self):
        raise NotImplementedError
