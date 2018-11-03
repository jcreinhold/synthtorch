#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.models.unet

holds the architecture for a 3d unet [1]

References:
    [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
        “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
        in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['Unet']

import logging
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from synthnn import get_act, get_norm

logger = logging.getLogger(__name__)


class Unet(torch.nn.Module):
    """
    defines a 3d unet [1] in pytorch

    Args:
        n_layers (int): number of layers (to go down and up)
        kernel_size (int): size of kernel (symmetric)
        dropout_p (int): dropout probability for each layer
        patch_size (int): dimension of one side of a cube (i.e., extracted "patch" is a patch_sz^3 size 3d-array)
        channel_base_power (int): 2 ** channel_base_power is the number of channels in the first layer
            and increases in each proceeding layer such that in the n-th layer there are
            2 ** channel_base_power + n channels (this follows the convention in [1])
        add_two_up (bool): flag to add two to the kernel size on the upsampling following
            the paper [2]
        normalization_layer (str): type of normalization layer to use (batch or [instance])
        activation (str): type of activation to use throughout network except final ([relu], lrelu, linear, sigmoid, tanh)
        output_activation (str): final activation in network (relu, lrelu, [linear], sigmoid, tanh)
        use_up_conv (bool): Use resize-convolution in the U-net as per the Distill article:
                            "Deconvolution and Checkerboard Artifacts" [Default=False]

    References:
        [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
            “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
            in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.
        [2] C. Zhao, A. Carass, J. Lee, Y. He, and J. L. Prince, “Whole Brain Segmentation and Labeling
            from CT Using Synthetic MR Images,” MLMI, vol. 10541, pp. 291–298, 2017.

    """
    def __init__(self, n_layers: int, kernel_size: int=3, dropout_p: float=0, patch_size: int=64, channel_base_power: int=5,
                 add_two_up: bool=False, normalization: str='instance', activation: str='relu', output_activation: str='linear',
                 use_up_conv: bool=False):
        super(Unet, self).__init__()
        # setup and store instance parameters
        self.n_layers = n_layers
        self.kernel_sz = kernel_size
        self.dropout_p = dropout_p
        self.patch_sz = patch_size
        self.channel_base_power = channel_base_power
        self.a2u = 2 if add_two_up else 0
        self.norm = nm = normalization
        self.act = a = activation
        self.out_act = oa = output_activation
        self.use_up_conv = use_up_conv
        def lc(n): return int(2 ** (channel_base_power + n))  # shortcut to layer count
        # define the model layers here to make them visible for autograd
        self.start = self.__dbl_conv_act(1, lc(0), lc(1), act=(a, a), norm=(nm, nm))
        self.down_layers = nn.ModuleList([self.__dbl_conv_act(lc(n), lc(n), lc(n + 1), act=(a, a), norm=(nm, nm))
                                          for n in range(1, n_layers)])
        self.bridge = self.__dbl_conv_act(lc(n_layers), lc(n_layers), lc(n_layers + 1), act=(a, a), norm=(nm, nm))
        self.up_layers = nn.ModuleList([self.__dbl_conv_act(lc(n) + lc(n - 1), lc(n - 1), lc(n - 1),
                                                            (kernel_size+self.a2u, kernel_size),
                                                            act=(a, a), norm=(nm, nm))
                                        for n in reversed(range(3, n_layers + 2))])
        self.finish = self.__final_conv(lc(2) + lc(1), lc(1), None, a, nm, oa)
        if use_up_conv:
            self.up_conv = nn.ModuleList([self.__conv(lc(n), lc(n)) for n in reversed(range(2, n_layers + 2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start(x)
        dout = [x]
        x = F.max_pool3d(dout[-1], (2, 2, 2))
        for dl in self.down_layers:
            dout.append(dl(x))
            x = F.max_pool3d(dout[-1], (2, 2, 2))
        x = F.interpolate(self.bridge(x), size=dout[-1].shape[2:])
        if self.use_up_conv:
            x = self.up_conv[0](x)
        for i, (ul, d) in enumerate(zip(self.up_layers, reversed(dout)), 1):
            x = ul(torch.cat((x, d), dim=1))
            x = F.interpolate(x, size=dout[-i-1].shape[2:])
            if self.use_up_conv:
                x = self.up_conv[i](x)
        x = self.finish(torch.cat((x, dout[0]), dim=1))
        return x

    def __conv(self, in_c: int, out_c: int, kernel_sz: Optional[int]=None) -> nn.Sequential:
        ksz = self.kernel_sz if kernel_sz is None else kernel_sz
        c = nn.Sequential(
                nn.ReplicationPad3d(ksz // 2),
                nn.Conv3d(in_c, out_c, ksz))
        return c

    def __conv_act(self, in_c: int, out_c: int, kernel_sz: Optional[int]=None,
                   act: Optional[str]=None, norm: Optional[str]=None) -> nn.Sequential:
        ksz = self.kernel_sz if kernel_sz is None else kernel_sz
        activation = get_act(act) if act is not None else get_act('relu')
        normalization = get_norm(norm, out_c) if norm is not None else get_norm('instance', out_c)
        ca = nn.Sequential(
            self.__conv(in_c, out_c, ksz),
            normalization,
            activation,
            nn.Dropout3d(self.dropout_p, inplace=True)) if normalization is not None else \
              nn.Sequential(
            self.__conv(in_c, out_c, ksz),
            activation,
            nn.Dropout3d(self.dropout_p, inplace=True))
        return ca

    def __dbl_conv_act(self, in_c: int, mid_c: int, out_c: int,
                       kernel_sz: Tuple[Optional[int], Optional[int]]=(None,None),
                       act: Tuple[Optional[str], Optional[str]]=(None,None),
                       norm: Tuple[Optional[str], Optional[str]]=(None,None)) -> nn.Sequential:
        dca = nn.Sequential(
            self.__conv_act(in_c, mid_c, kernel_sz[0], act[0], norm[0]),
            self.__conv_act(mid_c, out_c, kernel_sz[1], act[1], norm[1]))
        return dca

    def __final_conv(self, in_c: int, mid_c: int, kernel_sz: Optional[int]=None,
                     act: Optional[str]=None, norm: Optional[str]=None, out_act: Optional[str]=None):
        ca = self.__conv_act(in_c, mid_c, kernel_sz, act, norm)
        c = self.__conv(mid_c, 1, 1)
        fc = nn.Sequential(ca, c, get_act(out_act)) if out_act != 'linear' else nn.Sequential(ca, c)
        return fc