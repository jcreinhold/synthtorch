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

from ..learn import SelfAttention, SeparableConv2d, SeparableConv3d
from ..util import get_act, get_norm3d, get_norm2d, get_loss

logger = logging.getLogger(__name__)


class Unet(torch.nn.Module):
    """
    defines a 2d or 3d unet [1] in pytorch

    Args:
        n_layers (int): number of layers (to go down and up)
        kernel_size (int): size of kernel (symmetric)
        dropout_p (float): dropout probability for each layer
        channel_base_power (int): 2 ** channel_base_power is the number of channels in the first layer
            and increases in each proceeding layer such that in the n-th layer there are
            2 ** channel_base_power + n channels (this follows the convention in [1])
        add_two_up (bool): flag to add two to the kernel size on the upsampling following
            the paper [2]
        normalization_layer (str): type of normalization layer to use (batch or [instance])
        activation (str): type of activation to use throughout network except final ([relu], lrelu, linear, sigmoid, tanh)
        output_activation (str): final activation in network (relu, lrelu, [linear], sigmoid, tanh)
        is_3d (bool): if false define a 2d unet, otherwise the network is 3d
        interp_mode (str): use one of {'nearest', 'bilinear', 'trilinear'} for upsampling interpolation method
            depending on if the unet is 3d or 2d
        enable_dropout (bool): enable the use of spatial dropout (if dropout_p is set to zero then there will be no
            dropout, however if this is false and dropout_p > 0, then no dropout will be used) [Default=True]
        enable_bias (bool): enable bias calculation in final and upsampconv layers [Default=False]
        n_input (int): number of input channels to network [Default=1]
        n_output (int): number of output channels for network [Default=1]
        no_skip (bool): use no skip connections [Default=False]
        noise_lvl (float): add gaussian noise to weights with this std [Default=0]
        loss (str): loss function used to train network
        attention (bool): use (self-)attention gates (only works with 2d networks)
        inplace (bool): use inplace operations (for prediction)
        separable (bool): use separable convolutions instead of full convolutions
        softmax (bool): use a softmax before the final layer

    References:
        [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
            “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
            in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.
        [2] C. Zhao, A. Carass, J. Lee, Y. He, and J. L. Prince, “Whole Brain Segmentation and Labeling
            from CT Using Synthetic MR Images,” MLMI, vol. 10541, pp. 291–298, 2017.

    """
    def __init__(self, n_layers:int, kernel_size:int=3, dropout_p:float=0, channel_base_power:int=5,
                 add_two_up:bool=False, normalization:str='instance', activation:str='relu',
                 output_activation:str='linear', is_3d:bool=True, interp_mode:str='nearest', enable_dropout:bool=True,
                 enable_bias:bool=False, n_input:int=1, n_output:int=1, no_skip:bool=False, noise_lvl:float=0,
                 loss:Optional[str]=None, attention:bool=False, inplace:bool=False, separable:bool=False,
                 softmax:bool=False, **kwargs):
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
        self.interp_mode = interp_mode
        self.enable_dropout = enable_dropout
        self.enable_bias = enable_bias
        self.n_input = n_input
        self.n_output = n_output
        self.no_skip = no_skip
        self.noise_lvl = noise_lvl
        self.criterion = get_loss(loss)
        self.use_attention = attention and not is_3d
        self.separable = separable
        self.inplace = inplace
        self.softmax = softmax
        self.dropout_last = self.__class__.__name__ == 'Unet' or self.__class__.__name__ == 'OrdNet'
        nl = n_layers - 1
        def lc(n): return int(2 ** (channel_base_power + n))  # shortcut to layer channel count
        # define the model layers here to make them visible for autograd
        self.start = self._unet_blk(n_input, lc(0), lc(0), act=(a, a), norm=(nm, nm))
        self.down_layers = nn.ModuleList([self._unet_blk(lc(n), lc(n+1), lc(n+1), act=(a, a), norm=(nm, nm))
                                          for n in range(nl)])
        self.bridge = self._unet_blk(lc(nl), lc(nl+1), lc(nl+1), act=(a, a), norm=(nm, nm))
        self.up_layers = nn.ModuleList([self._unet_blk(lc(n) + lc(n) if not no_skip else lc(n),
                                                       lc(n), lc(n), (kernel_size+self.a2u, kernel_size),
                                                       act=(a, a), norm=(nm, nm))
                                        for n in reversed(range(1,nl+1))])
        self.finish = self._final(lc(0) + n_input if not no_skip else lc(0), n_output, oa, bias=enable_bias)
        self.upsampconvs = nn.ModuleList([self._upsampconv(lc(n+1), lc(n)) for n in reversed(range(nl+1))])
        if self.use_attention: self.attn = nn.ModuleList([SelfAttention(lc(n)) for n in reversed(range(1, nl+1))])

    def forward(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        x = self._fwd_skip(x, **kwargs) if not self.no_skip else self._fwd_no_skip(x, **kwargs)
        return x

    def _fwd_skip(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        x = self._fwd_skip_nf(x)
        x = self.finish(x)
        return x

    def _fwd_skip_nf(self, x:torch.Tensor) -> torch.Tensor:
        dout = [x]
        x = self._add_noise(self.start[0](x))
        dout.append(self._add_noise(self.start[1](x)))
        x = self._down(dout[-1])
        for dl in self.down_layers:
            for dli in dl: x = self._add_noise(dli(x))
            dout.append(x)
            x = self._down(dout[-1])
        x = self._add_noise(self.bridge[0](x))
        x = self.upsampconvs[0](self._add_noise(self._up(self.bridge[1](x), dout[-1].shape[2:])))
        for i, (ul, d) in enumerate(zip(self.up_layers, reversed(dout)), 1):
            if self.use_attention: d = self.attn[i-1](d)
            x = torch.cat((x, d), dim=1)
            x = self._add_noise(ul[0](x))
            x = self._add_noise(ul[1](x), (i == self.n_layers-1) and self.dropout_last)  # no dropout before 1x1
            x = self._up(x, dout[-i-1].shape[2:])  # doesn't do anything on the last iteration
            x = self.upsampconvs[i](x)
        if self.softmax: F.softmax(x, dim=1)
        x = torch.cat((x, dout[0]), dim=1)
        return x

    def _fwd_no_skip(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        x = self._fwd_no_skip_nf(x)
        x = self.finish(x)
        return x

    def _fwd_no_skip_nf(self, x:torch.Tensor) -> torch.Tensor:
        sz = [x.shape]
        for si in self.start: x = self._add_noise(si(x))
        x = self._down(x)
        for dl in self.down_layers:
            for dli in dl: x = self._add_noise(dli(x))
            sz.append(x.shape)
            x = self._down(x)
        x = self._add_noise(self.bridge[0](x))
        x = self.upsampconvs[0](self._add_noise(self._up(self.bridge[1](x), sz[-1][2:])))
        for i, (ul, s) in enumerate(zip(self.up_layers, reversed(sz)), 1):
            if self.use_attention: x = self.attn[i-1](x)
            x = self._add_noise(ul[0](x))
            x = self._add_noise(ul[1](x), (i == self.n_layers-1) and self.dropout_last)  # no dropout before 1x1
            x = self._up(x, sz[-i-1][2:])  # doesn't do anything on the last iteration
            x = self.upsampconvs[i](x)
        if self.softmax: F.softmax(x, dim=1)
        return x

    def _down(self, x:torch.Tensor) -> torch.Tensor:
        y = (F.max_pool3d(x, (2,2,2)) if self.is_3d else F.max_pool2d(x, (2,2)))
        return y

    def _up(self, x:torch.Tensor, sz:Union[Tuple[int,int,int], Tuple[int,int]]) -> torch.Tensor:
        y = F.interpolate(x, size=sz, mode=self.interp_mode)
        return y

    def _add_noise(self, x:torch.Tensor, skip:bool=False) -> torch.Tensor:
        if skip: return x
        if self.dropout_p > 0:
            x = F.dropout3d(x, self.dropout_p, training=self.enable_dropout, inplace=self.inplace) if self.is_3d else \
                F.dropout2d(x, self.dropout_p, training=self.enable_dropout, inplace=self.inplace)
        if self.noise_lvl > 0:
            x = x + (torch.randn_like(x.detach()) * self.noise_lvl)
        return x

    def _conv(self, in_c:int, out_c:int, kernel_sz:Optional[int]=None, bias:bool=False) -> nn.Sequential:
        ksz = self.kernel_sz if kernel_sz is None else kernel_sz
        bias = False if self.norm != 'none' and not bias else True
        if not self.separable or ksz == 1:
            layers = [nn.Conv3d(in_c, out_c, ksz, bias=bias)] if self.is_3d else \
                     [nn.Conv2d(in_c, out_c, ksz, bias=bias)]
        else:
            layers = [SeparableConv3d(in_c, out_c, ksz, bias=bias)] if self.is_3d else \
                     [SeparableConv2d(in_c, out_c, ksz, bias=bias)]
        if ksz > 1: layers = [nn.ReplicationPad3d(ksz // 2)] + layers if self.is_3d else \
                             [nn.ReflectionPad2d(ksz // 2)] + layers
        c = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
        return c

    def _conv_act(self, in_c:int, out_c:int, kernel_sz:Optional[int]=None,
                  act:Optional[str]=None, norm:Optional[str]=None) -> nn.Sequential:
        ksz = kernel_sz or self.kernel_sz
        activation = get_act(act) if act is not None else get_act('relu')
        layers = [self._conv(in_c, out_c, ksz)]
        if norm in [None, 'instance', 'batch', 'layer']:
             normalization = get_norm3d(norm, out_c) if norm is not None and self.is_3d else \
                             get_norm3d('instance', out_c) if self.is_3d else \
                             get_norm2d(norm, out_c) if norm is not None and not self.is_3d else \
                             get_norm2d('instance', out_c)
             if normalization is not None: layers.append(normalization)
        elif norm == 'weight':   layers[0][1] = nn.utils.weight_norm(layers[0][1])
        elif norm == 'spectral': layers[0][1] = nn.utils.spectral_norm(layers[0][1])
        layers.append(activation)
        ca = nn.Sequential(*layers)
        return ca

    def _unet_blk(self, in_c:int, mid_c:int, out_c:int,
                  kernel_sz:Tuple[Optional[int],Optional[int]]=(None,None),
                  act:Tuple[Optional[str],Optional[str]]=(None,None),
                  norm:Tuple[Optional[str],Optional[str]]=(None,None)) -> nn.Sequential:
        layers = [self._conv_act(in_c,  mid_c, kernel_sz[0], act[0], norm[0]),
                  self._conv_act(mid_c, out_c, kernel_sz[1], act[1], norm[1])]
        dca = nn.ModuleList(layers)
        return dca

    def _upsampconv(self, in_c:int, out_c:int):
        usc = self._conv(in_c, out_c, 3, bias=self.enable_bias)
        return usc

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        c = self._conv(in_c, out_c, 1, bias=bias)
        fc = nn.Sequential(c, get_act(out_act)) if out_act != 'linear' else c
        return fc

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward(x)
