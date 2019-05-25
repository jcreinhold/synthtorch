#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.learn.loss

define loss functions for neural network training

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 20, 2018
"""

__all__ = ['CosineProximityLoss',
           'SegAELoss',
           'VAELoss']

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..errors import SynthtorchError


class CosineProximityLoss(nn.Module):
    """ minimize the cosine proximity between an input and a target """
    def forward(self, y_hat:torch.Tensor, y:torch.Tensor):
        cp = torch.dot(y_hat.flatten(), y.flatten()) / (torch.norm(y_hat) * torch.norm(y))
        return 1 - cp


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, xhat, x):
        recon_x, mu, logvar = xhat
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
        return MSE + KLD


class SegAELoss(nn.Module):
    def __init__(self, ortho_penalty:float=1, norm_penalty:float=1, mse:bool=False, initialize:int=5, n_seg:int=4):
        super().__init__()
        self.op = ortho_penalty
        self.np = norm_penalty
        self.mse = mse
        self.initialize = initialize
        self.n_seg = n_seg
        try:
            from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
            from sklearn.mixture import GaussianMixture
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('SegAE requires scipy and scikit-learn')
        self.fill_holes = binary_fill_holes
        self.erode = binary_erosion
        self.gmm = GaussianMixture
        self.count = 0

    def calc_brain_mask(self, img:np.ndarray, t:float=0, fill_holes:bool=True, erosion_iter=1) -> np.ndarray:
        """ calculate a brain mask by thresholding (assume skull-stripped) """
        brain_mask = img > img.mean() if t is None else img > t
        if fill_holes: brain_mask = self.fill_holes(brain_mask)
        if erosion_iter > 0: self.erode(brain_mask, iterations=erosion_iter)
        return brain_mask

    def calc_tissue_mask(self, img:np.ndarray, brain_mask:np.ndarray) -> np.ndarray:
        """ calculate a tissue mask with a brain image """
        brain = np.expand_dims(img[brain_mask].flatten(), 1)
        gmm = self.gmm(self.n_seg)
        gmm.fit(brain)
        classes = np.argsort(gmm.means_.squeeze())
        tmp_predicted = gmm.predict(brain)
        predicted = np.zeros_like(tmp_predicted)
        for i, c in enumerate(classes):
            predicted[tmp_predicted == c] = i
        tissue_mask = np.zeros_like(img)
        tissue_mask[brain_mask] = predicted
        return tissue_mask

    @staticmethod
    def _cosine_proximity(x, y):
        return torch.dot(x, y)/(torch.norm(x)*torch.norm(y))

    @staticmethod
    def _norm_dist(x, y):
        """ less than or equal to one by the triangle inequality """
        return torch.abs(torch.norm(x) - torch.norm(y)) / torch.norm(x - y)

    def _ae_loss(self, x, y):
        loss = F.mse_loss(x**3, y**3) if self.mse else -self._cosine_proximity(x.flatten(), y.flatten())
        return loss

    def forward(self, yhat_seg, y):
        self.count += 1
        device = y.get_device() if y.is_cuda else 'cpu'
        yhat, seg = yhat_seg
        if self.count <= self.initialize:
            # assume first image is flair (so lesions are in own class), cube for better separability
            y_ = y[:,0,...].cpu().detach().numpy() ** 3
            bm = self.calc_brain_mask(y_)
            tm = self.calc_tissue_mask(y_, bm).astype(np.int64)
            loss = self._ae_loss(yhat, y) + self.op * F.cross_entropy(seg, torch.from_numpy(tm).to(device))
        else:
            C = seg.shape[1]
            if self.op > 0:
                ortho_penalty = torch.tensor([0.], requires_grad=True, dtype=y.dtype).to(device)
                for ci in range(C):
                    for cj in range(ci+1, C):
                        si, sj = seg[:,ci,...].flatten(), seg[:,cj,...].flatten()
                        ortho_penalty = ortho_penalty + self.op * torch.abs(self._cosine_proximity(si, sj))
                        if self.np > 0: ortho_penalty = ortho_penalty + self.np * self._norm_dist(si, sj)
            else:
                ortho_penalty = 0
            loss = self._ae_loss(yhat, y) + ortho_penalty
        return loss

