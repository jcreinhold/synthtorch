#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.models.vae

construct a variational autoencoder

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jan 29, 2019
"""

__all__ = ['VAE']

import logging
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .unet import Unet
from synthtorch.learn.loss import VAELoss

logger = logging.getLogger(__name__)


class VAE(Unet):
    def __init__(self, n_layers:int, img_dim:Union[Tuple[int,int],Tuple[int,int,int]],
                 channel_base_power:int=5, activation:str='relu', is_3d:bool=True,
                 n_input:int=1, n_output:int=1, latent_size=2048, **kwargs):
        super(VAE, self).__init__(n_layers, channel_base_power=channel_base_power, activation=activation,
                                  normalization='batch', is_3d=is_3d, enable_dropout=False, enable_bias=True,
                                  n_input=n_input, n_output=n_output, no_skip=True)
        del self.bridge
        self.sz = []
        self.latent_size = latent_size
        self.criterion = VAELoss()
        def lc(n): return int(2 ** (channel_base_power + n))  # shortcut to layer channel count
        self.lc = lc(n_layers)
        img_dim_conv = self.__test_encode(torch.zeros((1, n_input, *img_dim), dtype=torch.float32))
        self.fsz = img_dim_conv[1:]
        self.esz = int(np.prod(self.fsz))
        logger.debug(f'Size after Conv = {self.fsz}; Encoding size = {self.esz}')

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(self.esz, latent_size)
        self.fc_bn1 = nn.BatchNorm1d(latent_size)
        self.fc21 = nn.Linear(latent_size, latent_size)
        self.fc22 = nn.Linear(latent_size, latent_size)

        # Sampling vector
        self.fc3 = nn.Linear(latent_size, latent_size)
        self.fc_bn3 = nn.BatchNorm1d(latent_size)
        self.fc4 = nn.Linear(latent_size, self.esz)
        self.fc_bn4 = nn.BatchNorm1d(self.esz)

        # replace first upsampconv to not reduce channels
        self.upsampconvs[0] = self._conv(lc(n_layers-1), lc(n_layers-1), 3, bias=True)

    def encode(self, x):
        for si in self.start: x = si(x)
        x = self._down(x)
        for dl in self.down_layers:
            for dli in dl: x = dli(x)
            x = self._down(x)
        x = F.relu(self.fc_bn1(self.fc1(x.view(x.size(0), self.esz))))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    def __test_encode(self, x):
        self.sz.append(x.shape)
        for si in self.start: x = si(x)
        x = self._down(x)
        for dl in self.down_layers:
            for dli in dl: x = dli(x)
            self.sz.append(x.shape)
            x = self._down(x)
        return x.shape

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = F.relu(self.fc_bn3(self.fc3(z)))
        z = self.upsampconvs[0](self._up(F.relu(self.fc_bn4(self.fc4(z))).view(z.size(0), *self.fsz), self.sz[-1][2:]))
        for i, ul in enumerate(self.up_layers, 1):
            for uli in ul: z = uli(z)
            z = self._up(z, self.sz[-i-1][2:])
            z = self.upsampconvs[i](z)
        z = self.finish(z)
        return z

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def predict(self, x, *args, **kwargs):
        """ predict from a sample `x` """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)
