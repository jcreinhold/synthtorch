#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.models.vae

construct a variational autoencoder

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jan 29, 2019
"""

__all__ = ['VAE',
           'VAELoss']

import logging
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .unet import Unet

logger = logging.getLogger(__name__)


class VAE(Unet):
    def __init__(self, n_layers:int, img_dim:Union[Tuple[int,int],Tuple[int,int,int]],
                 channel_base_power:int=5, activation:str='relu', is_3d:bool=True,
                 n_input:int=1, n_output:int=1, latent_size=2048):
        super(VAE, self).__init__(n_layers, channel_base_power=channel_base_power, activation=activation,
                                  normalization='batch', is_3d=is_3d, enable_dropout=False, enable_bias=True,
                                  n_input=n_input, n_output=n_output, no_skip=True)
        del self.bridge, self.upsampconvs[0]
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

    def encode(self, x):
        x = self.start(x)
        x = self._down(x)
        for dl in self.down_layers:
            x = dl(x)
            x = self._down(x)
        x = F.relu(self.fc_bn1(self.fc1(x.view(x.size(0), self.esz))))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    def __test_encode(self, x):
        self.sz.append(x.shape)
        x = self.start(x)
        x = self._down(x)
        for dl in self.down_layers:
            x = dl(x)
            self.sz.append(x.shape)
            x = self._down(x)
        return x.shape

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = F.relu(self.fc_bn3(self.fc3(z)))
        z = self._up(F.relu(self.fc_bn4(self.fc4(z))).view(z.size(0), *self.fsz), self.sz[-1][2:])
        for i, ul in enumerate(self.up_layers, 1):
            z = ul(z)
            z = self._up(z, self.sz[-i-1][2:])
            z = self.upsampconvs[i-1](z)
        z = self.finish(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def predict(self, x):
        """ predict from a sample `x` """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x, recon_x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
        return MSE + KLD
