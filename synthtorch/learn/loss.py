#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.learn.loss

define loss functions for neural network training

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 20, 2018
"""

__all__ = ['CosineProximityLoss',
           'VAELoss']

import torch
from torch import nn


class CosineProximityLoss(nn.Module):
    """ minimize the cosine proximity between an input and a target """

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        cp = torch.dot(y_hat.flatten(), y.flatten()) / (torch.norm(y_hat) * torch.norm(y))
        return 1 - cp


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, xhat, x):
        recon_x, mu, logvar = xhat
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
