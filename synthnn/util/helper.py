#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.util.helper

define helper function for defining neural networks in pytorch

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['get_act',
           'get_norm']

from typing import Optional

from torch import nn

from ..errors import SynthNNError


def get_act(name: str, inplace: bool=True, params: Optional[dict]=None):
    """
    get activation module from pytorch
    must be one of: relu, lrelu, linear, tanh, sigmoid

    Args:
        name (str): name of activation function desired
        inplace (bool): flag activation to do operations in-place (if option available)
        params (dict): dictionary of parameters (as per pytorch documentation)

    Returns:
        act: instance of activation class
    """
    if name.lower() == 'relu':
        act = nn.ReLU(inplace=inplace)
    elif name.lower() == 'lrelu':
        act = nn.LeakyReLU(inplace=inplace) if params is None else nn.LeakyReLU(inplace=inplace, **params)
    elif name.lower() == 'linear':
        act = nn.LeakyReLU(1, inplace=inplace)  # hack to get linear output
    elif name.lower() == 'tanh':
        act = nn.Tanh()
    elif name.lower() == 'sigmoid':
        act = nn.Sigmoid()
    else:
        raise SynthNNError(f'Activation: "{name}" not a valid activation function or not supported.')
    return act


def get_norm(name: str, num_features: int, params: Optional[dict]=None):
    """
    get normalization module from pytorch
    must be one of: instance, batch, none

    Args:
        name (str): name of normalization function desired
        num_features (int): number of channels in the normalization layer
        params (dict): dictionary of optional other parameters for the normalization layer
            as specified by the pytorch documentation

    Returns:
        norm: instance of normalization layer
    """
    if name.lower() == 'instance':
        norm = nn.InstanceNorm3d(num_features, affine=True) if params is None else nn.InstanceNorm3d(num_features, **params)
    elif name.lower() == 'batch':
        norm = nn.BatchNorm3d(num_features) if params is None else nn.BatchNorm3d(num_features, **params)
    elif name.lower() == 'none':
        norm = None
    else:
        raise SynthNNError(f'Normalization: "{name}" not a valid normalization routine or not supported.')
    return norm
