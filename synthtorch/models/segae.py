#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.models.segae

implements a segmentation autoencoder as per [1]
with numerous modifications which were empirically found
to produce more robust results

References:
    [1] Atlason, H. E., Love, A., Sigurdsson, S., Gudnason, V., & Ellingsen, L. M. (2018).
        Unsupervised brain lesion segmentation from MRI using a convolutional autoencoder, 2–4.
        Retrieved from http://arxiv.org/abs/1811.09655

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Mar 1, 2019
"""

__all__ = ['SegAE']

import logging
from typing import List, Tuple

import torch
from torch import nn

from .unet import Unet
from synthtorch.learn.loss import SegAELoss

logger = logging.getLogger(__name__)


class SegAE(Unet):

    def __init__(self, n_layers:int=2, dropout_prob:float=0, channel_base_power:int=5, activation:str= 'lrelu',
                 is_3d:bool=True, enable_dropout:bool=True, n_input:int=1, n_output:int=1, inplace:bool=False,
                 n_seg:int=5, ortho_penalty:float=1, norm_penalty:float=1, use_mse:bool=False, no_skip:bool=True,
                 use_mask:bool=True, initialize:int=0, seg_min:float=0, freeze_last:bool=False,
                 last_init:List[float]=None, **kwargs):
        """ Implements heavily modified version of SegAE [1] """
        self.n_seg = n_seg
        self.ortho_penalty = ortho_penalty
        self.norm_penalty = norm_penalty
        self.use_mask = use_mask
        self.seg_min = seg_min
        self.freeze_last = freeze_last
        self.last_init = last_init
        super(SegAE, self).__init__(n_layers, dropout_prob=dropout_prob, channel_base_power=channel_base_power,
                                    activation=activation, normalization='batch', is_3d=is_3d,
                                    enable_dropout=enable_dropout, enable_bias=True, n_input=n_input, n_output=n_output,
                                    no_skip=no_skip, inplace=inplace)
        self.criterion = SegAELoss(ortho_penalty, norm_penalty, use_mse, initialize, n_seg)
        self.finish[2].weight.requires_grad = not freeze_last


    def _final(self, in_c:int, out_c:int, *args, **kwargs):
        f0 = self._conv_act(in_c, in_c//2, 1, act=self.act, norm=self.norm)
        f1 = [self._conv(in_c//2, self.n_seg, 1), nn.Softmax(1)]
        if self.seg_min > 0: f1.append(nn.Threshold(self.seg_min, 0))
        f2 = self._conv(self.n_seg, out_c, 1, bias=False)  # force no bias on last layer
        return nn.ModuleList([f0, nn.Sequential(*f1), f2])

    def forward(self, x:torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.finish[2].weight.data.clamp_(min=0)
        x = self._fwd_skip(x) if not self.no_skip else self._fwd_no_skip(x)
        return x

    def _mask(self, x:torch.Tensor):
        # use the first image to create the brain mask
        mask = (x[:,0,...] > 0).unsqueeze(1).type_as(x) if self.use_mask else \
               (x[:,0,...] == x[:,0,...]).unsqueeze(1).type_as(x)
        return mask

    def _fwd_skip(self, x:torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self._mask(x)
        # use the first image to create the brain mask
        x = self._fwd_skip_nf(x)
        x = self.finish[0](x)
        seg = (self.finish[1](x * mask) * mask)
        x = self.finish[2](seg) * mask
        return x, seg

    def _fwd_no_skip(self, x:torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """ essentially the `no skip` forward pass in the unet adding the segmentation layer """
        mask = self._mask(x)
        x = self._fwd_no_skip_nf(x)
        x = self.finish[0](x)
        seg = (self.finish[1](x * mask) * mask)
        x = self.finish[2](seg) * mask
        return x, seg

    def predict(self, x:torch.Tensor, return_seg:bool=False, **kwargs) -> torch.Tensor:
        """ predict from a sample `x` """
        yhat, seg = self.forward(x)
        return yhat if not return_seg else seg
