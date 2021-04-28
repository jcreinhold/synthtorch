#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.models.unet

holds the architecture for a 2d or 3d unet [1,2,3]

References:
    [1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox.
        "U-net: Convolutional networks for biomedical image segmentation."
        International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
    [2] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
        “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
        in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.
    [3] C. Zhao, A. Carass, J. Lee, Y. He, and J. L. Prince, “Whole Brain Segmentation and Labeling
        from CT Using Synthetic MR Images,” MLMI, vol. 10541, pp. 291–298, 2017.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['Unet']

import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from synthtorch.learn import ChannelAttention, SelfAttention, SeparableConv3d, SeparableConv2d, SeparableConv1d
from synthtorch.util import get_act, get_norm3d, get_norm2d, get_norm1d, get_loss

logger = logging.getLogger(__name__)


class Unet(torch.nn.Module):
    """
    defines a 2d or 3d unet [1,2,3] in pytorch

    Args:
        n_layers (int): number of layers (to go down and up)
        kernel_size (tuple): size of kernel
        dropout_p (float): dropout probability for each layer
        channel_base_power (int): 2 ** channel_base_power is the number of channels in the first layer
            and increases in each proceeding layer such that in the n-th layer there are
            2 ** channel_base_power + n channels (this follows the convention in [1])
        normalization_layer (str): type of normalization layer to use (batch or [instance])
        activation (str): type of activation to use throughout network except final ([relu], lrelu, linear, sigmoid, tanh)
        output_activation (str): final activation in network (relu, lrelu, [linear], sigmoid, tanh)
        dim (int): dimension of network (i.e., 1 for 1d, 2 for 2d, 3 for 3d)
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
        attention (str): use (self- or channelwise- )attention gates (self-attn only works with 2d networks)
        inplace (bool): use inplace operations (for prediction)
        separable (bool): use separable convolutions instead of full convolutions
        softmax (bool): use a softmax before the final layer
        input_connect (bool): use a skip connection from the input to near the end of the network
        all_conv (bool): use strided conv to downsample instead of max-pool and use pixelshuffle to upsample
        resblock (bool): use residual (addition) connections on unet blocks (only works if all_conv true)
            note: this is an activation-before-addition type residual connection (see Fig 4(c) in [4])
        semi_3d (int): use one or two (based on input, 1 or 2) 3d conv layers in an otherwise 2d network,
            should specify an oblong kernel shape, e.g., (3,3,1)
        affine (bool): use affine transform in normalization modules

    References:
        [1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox.
            "U-net: Convolutional networks for biomedical image segmentation."
            International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
        [2] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
            “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
            in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.
        [3] C. Zhao, A. Carass, J. Lee, Y. He, and J. L. Prince, “Whole Brain Segmentation and Labeling
            from CT Using Synthetic MR Images,” MLMI, vol. 10541, pp. 291–298, 2017.
        [4] He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision.
            Springer, Cham, 2016.
    """

    def __init__(self, n_layers: int, kernel_size: Tuple[int] = 3, dropout_prob: float = 0, channel_base_power: int = 5,
                 normalization: str = 'instance', activation: str = 'relu', output_activation: str = 'linear',
                 dim: int = 3, interp_mode: str = 'nearest', enable_dropout: bool = True,
                 enable_bias: bool = False, n_input: int = 1, n_output: int = 1, no_skip: bool = False,
                 noise_lvl: float = 0,
                 loss: Optional[str] = None, attention: Optional[str] = None, inplace: bool = False,
                 separable: bool = False,
                 softmax: bool = False, input_connect: bool = True, all_conv: bool = False, resblock: bool = False,
                 semi_3d: int = 0, affine: bool = True, **kwargs):
        super(Unet, self).__init__()
        # setup and store instance parameters
        self.n_layers = n_layers
        self.kernel_sz = tuple(kernel_size) if isinstance(kernel_size, (tuple, list)) else \
            tuple([kernel_size for _ in range(dim)])
        self.dropout_prob = dropout_prob
        self.channel_base_power = channel_base_power
        self.affine = affine
        self.norm = nm = normalization
        self.act = a = activation
        self.out_act = oa = output_activation
        self.dim = dim
        self.interp_mode = interp_mode
        self.enable_dropout = enable_dropout
        self.enable_bias = enable_bias
        self.n_input = n_input
        self.n_output = n_output
        self.no_skip = no_skip
        self.noise_lvl = noise_lvl
        self.loss = loss
        self.criterion = get_loss(loss)
        self.attention = attention
        self.separable = separable
        self.inplace = inplace
        self.softmax = softmax
        self.input_connect = input_connect
        self.all_conv = all_conv
        self.resblock = resblock and all_conv
        self.is_unet = self.__class__.__name__ == 'Unet'
        self.semi_3d = semi_3d if dim == 3 else 0
        nl = n_layers - 1

        def lc(n):
            return int(2 ** (channel_base_power + n))  # shortcut to layer channel count

        # define the model layers here to make them visible for autograd
        self.start = self._unet_blk(n_input if not self.semi_3d else lc(0), lc(0), lc(0), act=(a, a), norm=(nm, nm))
        self.down_layers = nn.ModuleList([self._unet_blk(lc(n + 1) if all_conv else lc(n), lc(n + 1), lc(n + 1),
                                                         act=(a, a), norm=(nm, nm))
                                          for n in range(nl)])
        self.bridge = self._unet_blk(lc((nl + 1) if all_conv else nl), lc(nl + 1), lc(nl + 1), act=(a, a),
                                     norm=(nm, nm))
        self.up_layers = nn.ModuleList([self._unet_blk(lc(n + 1) if not no_skip else lc(n),
                                                       lc(n), lc(n), (self.kernel_sz, self.kernel_sz),
                                                       act=(a, a), norm=(nm, nm))
                                        for n in reversed(range(1, nl + 1))])
        self.end = self._unet_blk(lc(1) if not no_skip else lc(0), lc(0), lc(0),
                                  act=(a, 'softmax' if softmax and self.is_unet else a), norm=(nm, nm))
        self.finish = self._final(lc(0) + n_input if not no_skip and input_connect else lc(0), n_output,
                                  oa, bias=enable_bias)
        self.upsampconvs = nn.ModuleList([self._upsampconv(lc(n + 1), lc(n)) for n in reversed(range(nl + 1))])
        if self.attention is not None:
            i, is_ch = ((0, 1) if no_skip else (1, 2)), attention == 'channel'
            self.attn = nn.ModuleList([ChannelAttention(lc(n)) for n in reversed(range(i[0], nl + i[1]))]) if is_ch else \
                nn.ModuleList([SelfAttention(lc(n)) for n in reversed(range(nl + 1))])
        if self.all_conv: self.downsampconvs = nn.ModuleList(
            [self._downsampconv(lc(n), lc(n + 1)) for n in range(nl + 1)])
        if self.semi_3d > 0: self.init_conv = self._conv_act(n_input, lc(0), (3, 3, 3), a, nm)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self._fwd_skip(x, **kwargs) if not self.no_skip else self._fwd_no_skip(x, **kwargs)
        return x

    def _fwd_skip(self, x: torch.Tensor, **kwargs):
        x = self._fwd_skip_nf(x)
        x = self._finish(x)
        return x

    def _fwd_skip_nf(self, x: torch.Tensor) -> torch.Tensor:
        dout = [x] if self.input_connect else [None]
        if self.semi_3d > 0: x = self._add_noise(self.init_conv(x))
        x = self._add_noise(self.start[0](x))
        dout.append(self._add_noise(self.start[1](x)))
        x = self._down(dout[-1], 0)
        if self.all_conv: x = self._add_noise(x)
        for i, dl in enumerate(self.down_layers, 1):
            if self.resblock: xr = x
            for dli in dl: x = self._add_noise(dli(x))
            dout.append((x + xr) if self.resblock else x)
            x = self._down(dout[-1], i)
            if self.all_conv: x = self._add_noise(x)
        x = self._add_noise(self.bridge[0](x))
        x = self._up(self._add_noise(self.bridge[1](x)), dout[-1].shape[2:], 0)
        if self.all_conv: x = self._add_noise(x)
        for i, (ul, d) in enumerate(zip(self.up_layers, reversed(dout)), 1):
            if self.attention == 'self': d = self.attn[i - 1](d)
            if self.resblock: xr = x
            x = torch.cat((x, d), dim=1)
            if self.attention == 'channel': x = self.attn[i - 1](x)
            for uli in ul: x = self._add_noise(uli(x))
            x = self._up((x + xr) if self.resblock else x, dout[-i - 1].shape[2:], i)
            if self.all_conv: x = self._add_noise(x)
        if self.attention == 'self': dout[1] = self.attn[-1](dout[1])
        if self.resblock: xr = x
        x = torch.cat((x, dout[1]), dim=1)
        if self.attention == 'channel': x = self.attn[-1](x)
        for eli in self.end: x = self._add_noise(eli(x))
        if self.resblock: x = x + xr
        if self.input_connect: x = torch.cat((x, dout[0]), dim=1)
        return x

    def _fwd_no_skip(self, x: torch.Tensor, **kwargs):
        x = self._fwd_no_skip_nf(x)
        x = self._finish(x)
        return x

    def _fwd_no_skip_nf(self, x: torch.Tensor) -> torch.Tensor:
        sz = [x.shape]
        if self.semi_3d > 0: x = self._add_noise(self.init_conv(x))
        for si in self.start: x = self._add_noise(si(x))
        x = self._down(x, 0)
        if self.all_conv: x = self._add_noise(x)
        for i, dl in enumerate(self.down_layers, 1):
            if self.resblock: xr = x
            for dli in dl: x = self._add_noise(dli(x))
            sz.append(x.shape)
            x = self._down((x + xr) if self.resblock else x, i)
            if self.all_conv: x = self._add_noise(x)
        x = self._add_noise(self.bridge[0](x))
        x = self._up(self._add_noise(self.bridge[1](x)), sz[-1][2:], 0)
        if self.all_conv: x = self._add_noise(x)
        for i, (ul, s) in enumerate(zip(self.up_layers, reversed(sz)), 1):
            if self.attention is not None: x = self.attn[i - 1](x)
            if self.resblock: xr = x
            for uli in ul: x = self._add_noise(uli(x))
            x = self._up((x + xr) if self.resblock else x, sz[-i - 1][2:], i)
            if self.all_conv: x = self._add_noise(x)
        if self.attention is not None: x = self.attn[-1](x)
        if self.resblock: xr = x
        for eli in self.end: x = self._add_noise(eli(x))
        if self.resblock: x = x + xr
        return x

    def _finish(self, x: torch.Tensor):
        return self.finish(x)

    def _down(self, x: torch.Tensor, i: int) -> torch.Tensor:
        if not self.all_conv:
            ksz = [(2 if ks != 1 else 1) for ks in self.kernel_sz]
            return F.max_pool3d(x, ksz) if self.dim == 3 else \
                F.max_pool2d(x, ksz) if self.dim == 2 else \
                    F.max_pool1d(x, ksz)
        else:
            return self.downsampconvs[i](x)

    def _up(self, x: torch.Tensor, sz: Union[Tuple[int, int, int], Tuple[int, int]], i: int) -> torch.Tensor:
        if not self.all_conv or self.dim != 2: x = F.interpolate(x, size=sz, mode=self.interp_mode)
        x = self.upsampconvs[i](x)
        if self.all_conv and self.dim == 2 and x.shape[2:] != sz: x = F.interpolate(x, sz, mode=self.interp_mode)
        return x

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout_prob > 0:
            x = F.dropout3d(x, self.dropout_prob, training=self.enable_dropout,
                            inplace=self.inplace) if self.dim == 3 else \
                F.dropout2d(x, self.dropout_prob, training=self.enable_dropout,
                            inplace=self.inplace) if self.dim == 2 else \
                    F.dropout(x, self.dropout_prob, training=self.enable_dropout, inplace=self.inplace)
        if self.noise_lvl > 0:
            x = x + (torch.randn_like(x.detach()) * self.noise_lvl)
        return x

    def _conv(self, in_c: int, out_c: int, kernel_sz: Optional[Tuple[int]] = None, bias: bool = False,
              seq: bool = True, stride: Tuple[int] = None):
        ksz = kernel_sz or self.kernel_sz
        bias = False if self.norm != 'none' and not bias else True
        stride = stride or tuple([1 for _ in ksz])
        if not self.separable or all([ks == 1 for ks in ksz]):
            layers = [nn.Conv3d(in_c, out_c, ksz, bias=bias, stride=stride)] if self.dim == 3 else \
                [nn.Conv2d(in_c, out_c, ksz, bias=bias, stride=stride)] if self.dim == 2 else \
                    [nn.Conv1d(in_c, out_c, ksz, bias=bias, stride=stride)]
        else:
            layers = [SeparableConv3d(in_c, out_c, ksz, bias=bias, stride=stride)] if self.dim == 3 else \
                [SeparableConv2d(in_c, out_c, ksz, bias=bias, stride=stride)] if self.dim == 2 else \
                    [SeparableConv1d(in_c, out_c, ksz, bias=bias, stride=stride)]
        if any([ks > 1 for ks in ksz]):
            rp = tuple([ks // 2 for p in zip(reversed(ksz), reversed(ksz)) for ks in p])
            layers = [nn.ReplicationPad3d(rp)] + layers if self.dim == 3 else \
                [nn.ReflectionPad2d(rp)] + layers if self.dim == 2 else \
                    [nn.ReflectionPad1d(rp)] + layers
        if seq and len(layers) > 1:
            c = nn.Sequential(*layers)
        else:
            c = layers if len(layers) > 1 else layers[0]
        return c

    def _conv_act(self, in_c: int, out_c: int, kernel_sz: Optional[Tuple[int]] = None,
                  act: str = 'relu', norm: str = 'instance', seq: bool = True, stride: Tuple[int] = None):
        ksz = kernel_sz or self.kernel_sz
        activation = get_act(act)
        c = self._conv(in_c, out_c, ksz, seq=False, stride=stride)
        i = 1 if isinstance(c, list) else 0
        layers = [*c] if i == 1 else [c]
        if norm in ['instance', 'batch', 'layer']:
            layers.append(get_norm3d(norm, out_c, self.affine) if self.dim == 3 else \
                              get_norm2d(norm, out_c, self.affine) if self.dim == 2 else \
                                  get_norm1d(norm, out_c, self.affine))
        elif norm == 'weight':
            layers[i] = nn.utils.weight_norm(layers[i])
        elif norm == 'spectral':
            layers[i] = nn.utils.spectral_norm(layers[i])
        layers.append(activation)
        ca = nn.Sequential(*layers) if seq else layers
        return ca

    def _unet_blk(self, in_c: int, mid_c: int, out_c: int,
                  kernel_sz: Tuple[Optional[Tuple[int]], Optional[Tuple[int]]] = (None, None),
                  act: Tuple[Optional[str], Optional[str]] = (None, None),
                  norm: Tuple[Optional[str], Optional[str]] = (None, None)) -> nn.ModuleList:
        layers = [self._conv_act(in_c, mid_c, kernel_sz[0], act[0], norm[0]),
                  self._conv_act(mid_c, out_c, kernel_sz[1], act[1], norm[1])]
        dca = nn.ModuleList(layers)
        return dca

    def _upsampconv(self, in_c: int, out_c: int, scale: int = 2):
        ksz = tuple([(5 if ks != 1 else 1) for ks in self.kernel_sz])
        usc = self._conv(in_c, out_c, ksz, bias=self.enable_bias) if not self.all_conv else \
            self._conv_act(in_c, out_c, ksz, self.act, self.norm) if self.all_conv and self.dim != 2 else \
                nn.Sequential(*self._conv_act(in_c, out_c * (scale ** 2), (1, 1), self.act, self.norm, seq=False),
                              nn.PixelShuffle(scale))
        return usc

    def _downsampconv(self, in_c: int, out_c: int):
        stride = tuple([(2 if ks != 1 else 1) for ks in self.kernel_sz])
        dsc = self._conv_act(in_c, out_c, self.kernel_sz, self.act, self.norm, stride=stride)
        return dsc

    def _final(self, in_c: int, out_c: int, out_act: Optional[str] = None, bias: bool = False):
        ksz = tuple([1 for _ in self.kernel_sz])
        c = self._conv(in_c, out_c, ksz, bias=bias)
        if self.semi_3d != 2 and self.attention == 'channel':
            c = nn.Sequential(ChannelAttention(in_c), c)
        if self.semi_3d == 2:
            cs = [self._conv_act(in_c, in_c, (3, 3, 3), self.act, self.norm), c]
            if self.attention == 'channel': cs.insert(1, ChannelAttention(in_c))
            c = nn.Sequential(*cs)
        fc = nn.Sequential(c, get_act(out_act)) if out_act != 'linear' else c
        return fc

    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward(x)

    def freeze(self):
        """ freeze all but final layer """
        for p in self.start.parameters(): p.requires_grad = False
        for p in self.down_layers.parameters(): p.requires_grad = False
        for p in self.bridge.parameters(): p.requires_grad = False
        for p in self.up_layers.parameters(): p.requires_grad = False
        for p in self.end.parameters(): p.requires_grad = False
        for p in self.upsampconvs.parameters(): p.requires_grad = False
        if self.attention is not None:
            for p in self.attn.parameters(): p.requires_grad = False
        if self.all_conv:
            for p in self.downsampconvs.parameters(): p.requires_grad = False
        if self.semi_3d > 0:
            for p in self.init_conv.parameters(): p.requires_grad = False
