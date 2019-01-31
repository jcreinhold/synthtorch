#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.models.unet

holds the architecture for a 2d or 3d unet [1]

References:
    [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
        “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
        in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['Unet']

import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from synthnn import get_act, get_norm3d, get_norm2d, SynthNNError

logger = logging.getLogger(__name__)


class Unet(torch.nn.Module):
    """
    defines a 2d or 3d unet [1] in pytorch

    Args:
        n_layers (int): number of layers (to go down and up)
        kernel_size (int): size of kernel (symmetric)
        dropout_p (int): dropout probability for each layer
        channel_base_power (int): 2 ** channel_base_power is the number of channels in the first layer
            and increases in each proceeding layer such that in the n-th layer there are
            2 ** channel_base_power + n channels (this follows the convention in [1])
        add_two_up (bool): flag to add two to the kernel size on the upsampling following
            the paper [2]
        normalization_layer (str): type of normalization layer to use (batch or [instance])
        activation (str): type of activation to use throughout network except final ([relu], lrelu, linear, sigmoid, tanh)
        output_activation (str): final activation in network (relu, lrelu, [linear], sigmoid, tanh)
        is_3d (bool): if false define a 2d unet, otherwise the network is 3d
        deconv (bool): use transpose conv for upsampling and strided conv for down (instead of upsamp with interp)
        interp_mode (str): when using interpolation for upsampling (i.e., deconv==False), use one of
            {'nearest', 'bilinear', 'trilinear'} depending on if the unet is 3d or 2d
        upsampconv (bool): Use resize-convolution in the U-net as per the Distill article:
                            "Deconvolution and Checkerboard Artifacts" [Default=False]
        enable_dropout (bool): enable the use of dropout (if dropout_p is set to zero then there will be no dropout,
            however if this is false and dropout_p > 0, then no dropout will be used) [Default=True]
        enable_bias (bool): enable bias calculation in final and upsampconv layers [Default=False]
        n_input (int): number of input channels to network [Default=1]
        n_output (int): number of output channels for network [Default=1]
        no_skip (bool): use no skip connections [Default=False]

    References:
        [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
            “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
            in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.
        [2] C. Zhao, A. Carass, J. Lee, Y. He, and J. L. Prince, “Whole Brain Segmentation and Labeling
            from CT Using Synthetic MR Images,” MLMI, vol. 10541, pp. 291–298, 2017.

    """
    def __init__(self, n_layers:int, kernel_size:int=3, dropout_p:float=0, channel_base_power:int=5,
                 add_two_up:bool=False, normalization:str='instance', activation:str='relu', output_activation:str='linear',
                 is_3d:bool=True, deconv:bool=True, interp_mode:str='nearest', upsampconv:bool=False, enable_dropout:bool=True,
                 enable_bias:bool=False, n_input:int=1, n_output:int=1, no_skip=False):
        super(Unet, self).__init__()
        # setup and store instance parameters
        self.n_layers = n_layers
        self.kernel_sz = kernel_size
        self.dropout_p = dropout_p
        self.channel_base_power = channel_base_power
        self.a2u = 2 if add_two_up else 0
        self.norm = nm = normalization
        self.act = a = activation
        self.out_act = oa = output_activation
        self.is_3d = is_3d
        self.deconv = deconv
        self.interp_mode = interp_mode
        self.upsampconv = upsampconv
        self.enable_dropout = enable_dropout
        self.enable_bias = enable_bias
        self.n_input = n_input
        self.n_output = n_output
        self.no_skip = no_skip
        self.criterion = nn.MSELoss()
        def lc(n): return int(2 ** (channel_base_power + n))  # shortcut to layer channel count
        # define the model layers here to make them visible for autograd
        self.start = self._unet_blk(n_input, lc(0), lc(1), act=(a, a), norm=(nm, nm))
        self.down_layers = nn.ModuleList([self._unet_blk(lc(n), lc(n), lc(n+1), act=(a, a), norm=(nm, nm))
                                          for n in range(1, n_layers)])
        self.bridge = self._unet_blk(lc(n_layers), lc(n_layers), lc(n_layers+1), act=(a, a), norm=(nm, nm))
        self.up_layers = nn.ModuleList([self._unet_blk(lc(n) + lc(n-1) if not no_skip else lc(n), lc(n-1), lc(n-1),
                                                       (kernel_size+self.a2u, kernel_size),
                                                       act=(a, a), norm=(nm, nm))
                                        for n in reversed(range(3, n_layers+2))])
        self.finish = self._final_conv(lc(2) + n_input if not no_skip else lc(2), n_output, oa, bias=enable_bias)
        if upsampconv:
            self.upsampconvs = nn.ModuleList([self._conv(lc(n), lc(n), 3, bias=enable_bias)
                                              for n in reversed(range(2, n_layers+1))])
        if deconv:
            self.downconv = nn.ModuleList([self._conv_act(lc(n), lc(n), act=a, norm=nm, mode='down')
                                           for n in range(1, n_layers+1)])
            self.upconv = nn.ModuleList([self._conv_act(lc(n), lc(n), 2, act=a, norm=nm, mode='up')
                                         for n in reversed(range(2, n_layers+2))])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self._fwd_skip(x) if not self.no_skip else self._fwd_no_skip(x)
        return x

    def _fwd_skip(self, x:torch.Tensor) -> torch.Tensor:
        dout = [x]
        dout.append(self.start(x))
        x = self._down(dout[-1], 0)
        for i, dl in enumerate(self.down_layers, 1):
            dout.append(dl(x))
            x = self._dropout(self._down(dout[-1], i))
        x = self._dropout(self._up(self.bridge(x), dout[-1].shape[2:], 0))
        for i, (ul, d) in enumerate(zip(self.up_layers, reversed(dout)), 1):
            x = ul(torch.cat((x, d), dim=1))
            x = self._dropout(self._up(x, dout[-i-1].shape[2:], i))
            if self.upsampconv: x = self.upsampconvs[i-1](x)
        x = self.finish(torch.cat((x, dout[0]), dim=1))
        return x

    def _fwd_no_skip(self, x:torch.Tensor) -> torch.Tensor:
        sz = [x.shape]
        x = self.start(x)
        x = self._down(x, 0)
        for i, dl in enumerate(self.down_layers, 1):
            x = dl(x)
            sz.append(x.shape)
            x = self._dropout(self._down(x, i))
        x = self._dropout(self._up(self.bridge(x), sz[-1][2:], 0))
        for i, (ul, s) in enumerate(zip(self.up_layers, reversed(sz)), 1):
            x = ul(x)
            x = self._dropout(self._up(x, sz[-i-1][2:], i))
            if self.upsampconv: x = self.upsampconvs[i-1](x)
        x = self.finish(x)
        return x

    def _down(self, x:torch.Tensor, i:int) -> torch.Tensor:
        y = (F.max_pool3d(x, (2,2,2)) if self.is_3d else F.max_pool2d(x, (2,2))) if not self.deconv else self.downconv[i](x)
        return y

    def _up(self, x:torch.Tensor, sz:Union[Tuple[int,int,int], Tuple[int,int]], i:int) -> torch.Tensor:
        y = F.interpolate(x, size=sz, mode=self.interp_mode) if not self.deconv else self.upconv[i](x)
        return y

    def _dropout(self, x:torch.Tensor) -> torch.Tensor:
        x = F.dropout3d(x, self.dropout_p, training=self.enable_dropout) if self.is_3d else \
            F.dropout2d(x, self.dropout_p, training=self.enable_dropout)
        return x

    def _conv(self, in_c:int, out_c:int, kernel_sz:Optional[int]=None, mode:str=None, bias:bool=False) -> nn.Sequential:
        ksz = self.kernel_sz if kernel_sz is None else kernel_sz
        stride = 1 if mode is None else 2
        bias = False if self.norm != 'none' and not bias else True
        if mode is None or mode == 'down':
            c = nn.Sequential(nn.ReplicationPad3d(ksz // 2), nn.Conv3d(in_c, out_c, ksz, stride=stride, bias=bias)) if self.is_3d else \
                nn.Sequential(nn.ReflectionPad2d(ksz // 2),  nn.Conv2d(in_c, out_c, ksz, stride=stride, bias=bias))
        elif mode == 'up':
            c = nn.Sequential(nn.ConvTranspose3d(in_c, out_c, ksz, stride=stride, bias=bias)) if self.is_3d else \
                nn.Sequential(nn.ConvTranspose2d(in_c, out_c, ksz, stride=stride, bias=bias))
        else:
            raise SynthNNError(f'{mode} invalid, must be one of {{downconv, upconv}}')
        return c

    def _conv_act(self, in_c:int, out_c:int, kernel_sz:Optional[int]=None,
                  act:Optional[str]=None, norm:Optional[str]=None, mode:str=None) -> nn.Sequential:
        ksz = self.kernel_sz if kernel_sz is None else kernel_sz
        activation = get_act(act) if act is not None else get_act('relu')
        normalization = get_norm3d(norm, out_c) if norm is not None and self.is_3d else get_norm3d('instance', out_c) if self.is_3d else \
                        get_norm2d(norm, out_c) if norm is not None and not self.is_3d else get_norm2d('instance', out_c)
        layers = [self._conv(in_c, out_c, ksz, mode)]
        if normalization is not None: layers.append(normalization)
        layers.append(activation)
        ca = nn.Sequential(*layers)
        return ca

    def _unet_blk(self, in_c:int, mid_c:int, out_c:int,
                  kernel_sz:Tuple[Optional[int],Optional[int]]=(None,None),
                  act:Tuple[Optional[str],Optional[str]]=(None,None),
                  norm:Tuple[Optional[str],Optional[str]]=(None,None)) -> nn.Sequential:
        dca = nn.Sequential(
            self._conv_act(in_c,  mid_c, kernel_sz[0], act[0], norm[0]),
            self._conv_act(mid_c, out_c, kernel_sz[1], act[1], norm[1]))
        return dca

    def _final_conv(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        c = self._conv(in_c, out_c, 1, bias=bias)
        fc = nn.Sequential(c, get_act(out_act)) if out_act != 'linear' else nn.Sequential(c)
        return fc

    predict = forward
