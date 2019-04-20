#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.learn.layers

define auxillary layers for defining neural networks in pytorch

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 21, 2018
"""

__all__ = ['SelfAttention',
           'SeparableConv2d',
           'SeparableConv3d']

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


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.sep_conv = nn.Sequential(conv, pointwise)

    def forward(self, x):
        return self.sep_conv(x)


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv3d, self).__init__()
        conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.sep_conv = nn.Sequential(conv, pointwise)

    def forward(self, x):
        return self.sep_conv(x)
