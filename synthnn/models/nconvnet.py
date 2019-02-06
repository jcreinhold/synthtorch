#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.models.nconvnet

define the class for a N layer CNN with
no max pool, increase in channels, or any of that
fancy stuff. This is generally used for testing
purposes

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['SimpleConvNet']

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SimpleConvNet(torch.nn.Module):
    def __init__(self, n_layers:int, n_input:int=1, n_output:int=1, kernel_size:int=3, dropout_p:float=0, is_3d:bool=True):
        super(SimpleConvNet, self).__init__()
        self.n_layers = n_layers
        self.n_input = n_input
        self.n_output = n_output
        self.kernel_sz = kernel_size
        self.dropout_p = dropout_p
        self.is_3d = is_3d
        self.criterion = nn.MSELoss()
        if isinstance(kernel_size, int):
            self.kernel_sz = [kernel_size for _ in range(n_layers)]
        else:
            self.kernel_sz = kernel_size
        self.layers = nn.ModuleList([nn.Sequential(
            nn.ReplicationPad3d(ksz//2) if is_3d else nn.ReplicationPad2d(ksz//2),
            nn.Conv3d(n_input, n_output, ksz) if is_3d else nn.Conv2d(n_input, n_output, ksz),
            nn.ReLU(),
            nn.InstanceNorm3d(n_output, affine=True) if is_3d else nn.InstanceNorm2d(n_output, affine=True),
            nn.Dropout3d(dropout_p) if is_3d else nn.Dropout2d(dropout_p)) for ksz in self.kernel_sz])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward(x)
