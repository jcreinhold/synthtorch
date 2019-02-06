#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.learn.layers

define auxillary layers for defining neural networks in pytorch

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 21, 2018
"""

__all__ = ['SelfAttention']

import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """ Self attention layer for 2d (implementation inspired by fastai library) """
    def __init__(self, n_channels:int):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv1d(n_channels, n_channels//8, 1))
        self.key   = nn.utils.spectral_norm(nn.Conv1d(n_channels, n_channels//8, 1))
        self.value = nn.utils.spectral_norm(nn.Conv1d(n_channels, n_channels, 1))
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
