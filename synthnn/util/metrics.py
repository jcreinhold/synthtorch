#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.util.metrics

custom metrics for the synthnn package (used in fastai implementation)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 15, 2018
"""

__all__ = ['ncc',
           'mssim2d',
           'mssim3d',
           'mi']

import numpy as np
import torch

import synthqc


def ncc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ normalized cross-correlation """
    nccval = 0
    N = x.shape[0]
    for i in range(N):
        nccval += synthqc.normalized_cross_correlation(x[i,0,...], y[i,0,...])
    nccval /= N
    return nccval


def mssim2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ mean structural similarity (for 2d) """
    x = np.array(x.squeeze().transpose(0,2))
    y = np.array(y.squeeze().transpose(0,2))
    ms = synthqc.mssim(x, y, multichannel=True)
    ms = torch.Tensor(np.array([ms], dtype=np.float32)).squeeze()
    return ms


def mssim3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ mean structural similarity (for 3d) """
    x = np.array(x.squeeze())
    y = np.array(y.squeeze())
    N = x.shape[0]
    ms = 0
    for i in range(N):
        ms += synthqc.mssim(x[i], y[i])
    ms = torch.Tensor(np.array([ms/N], dtype=np.float32)).squeeze()
    return ms


def mi(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ mutual information """
    mival = 0
    N = x.shape[0]
    for i in range(N):
        mival += synthqc.mutual_info(np.array(x[i,0,...]).flatten(),
                                     np.array(y[i,0,...]).flatten())
    mival /= N
    mival = torch.Tensor(np.array([mival], dtype=np.float32)).squeeze()
    return mival

