#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.util.helper

define helper function for defining neural networks in pytorch

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['get_act',
           'get_loss',
           'get_norm1d',
           'get_norm2d',
           'get_norm3d',
           'get_optim',
           'init_weights']

from typing import Optional

import logging

import numpy as np
import torch
from torch import nn

from ..errors import SynthtorchError
from ..learn.loss import CosineProximityLoss
from ..learn.layers import Swish

logger = logging.getLogger(__name__)


def get_act(name:str, inplace:bool=True, params:Optional[dict]=None) -> nn.Module:
    """
    get activation module from pytorch
    must be one of: relu, lrelu, linear, tanh, sigmoid

    Args:
        name (str): name of activation function desired
        inplace (bool): flag activation to do operations in-place (if option available)
        params (dict): dictionary of parameters (as per pytorch documentation)

    Returns:
        act (activation): instance of activation class
    """
    if name.lower() == 'relu':
        act = nn.ReLU(inplace=inplace)
    elif name.lower() == 'lrelu':
        act = nn.LeakyReLU(inplace=inplace) if params is None else nn.LeakyReLU(inplace=inplace, **params)
    elif name.lower() == 'prelu':
        act = nn.PReLU() if params is None else nn.PReLU(**params)
    elif name.lower() == 'elu':
        act = nn.ELU(inplace=inplace) if params is None else nn.ELU(inplace=inplace, **params)
    elif name.lower() == 'celu':
        act = nn.CELU(inplace=inplace) if params is None else nn.CELU(inplace=inplace, **params)
    elif name.lower() == 'selu':
        act = nn.SELU(inplace=inplace)
    elif name.lower() == 'linear':
        act = nn.LeakyReLU(1, inplace=inplace)  # hack to get linear output
    elif name.lower() == 'tanh':
        act = nn.Tanh()
    elif name.lower() == 'sigmoid':
        act = nn.Sigmoid()
    elif name.lower() == 'softmax':
        act = nn.Softmax(dim=1)
    elif name.lower() == 'swish':
        act = Swish(inplace=inplace)
    else:
        raise SynthtorchError(f'Activation: "{name}" not a valid activation function or not supported.')
    return act


def get_norm1d(name:str, num_features:int, params:Optional[dict]=None) -> nn.Module:
    """
    get a 1d normalization module from pytorch
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
        norm = nn.InstanceNorm1d(num_features, affine=True) if params is None else nn.InstanceNorm1d(num_features, **params)
    elif name.lower() == 'batch':
        norm = nn.BatchNorm1d(num_features) if params is None else nn.BatchNorm1d(num_features, **params)
    elif name.lower() == 'layer':
        norm = nn.GroupNorm(1, num_features)
    elif name.lower() == 'none':
        norm = None
    else:
        raise SynthtorchError(f'Normalization: "{name}" not a valid normalization routine or not supported.')
    return norm


def get_norm2d(name:str, num_features:int, params:Optional[dict]=None) -> nn.Module:
    """
    get a 2d normalization module from pytorch
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
        norm = nn.InstanceNorm2d(num_features, affine=True) if params is None else nn.InstanceNorm2d(num_features, **params)
    elif name.lower() == 'batch':
        norm = nn.BatchNorm2d(num_features) if params is None else nn.BatchNorm2d(num_features, **params)
    elif name.lower() == 'layer':
        norm = nn.GroupNorm(1, num_features)
    elif name.lower() == 'none':
        norm = None
    else:
        raise SynthtorchError(f'Normalization: "{name}" not a valid normalization routine or not supported.')
    return norm


def get_norm3d(name:str, num_features:int, params:Optional[dict]=None) -> nn.Module:
    """
    get a 3d normalization module from pytorch
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
    elif name.lower() == 'layer':
        norm = nn.GroupNorm(1, num_features)
    elif name.lower() == 'none':
        norm = None
    else:
        raise SynthtorchError(f'Normalization: "{name}" not a valid normalization routine or not supported.')
    return norm


def get_optim(name:str):
    """ get an optimizer by name """
    if name.lower() == 'adam':
        optimizer = torch.optim.Adam
    elif name.lower() == 'adamw':
        optimizer = torch.optim.AdamW
    elif name.lower() == 'sgd':
        optimizer = torch.optim.SGD
    elif name.lower() == 'sgdw':
        from ..learn.optim import SGDW
        optimizer = SGDW
    elif name.lower() == 'nsgd':
        from ..learn.optim import NesterovSGD
        optimizer = NesterovSGD
    elif name.lower() == 'nsgdw':
        from ..learn.optim import NesterovSGDW
        optimizer = NesterovSGDW
    elif name.lower() == 'rmsprop':
        optimizer = torch.optim.rmsprop
    elif name.lower() == 'adagrad':
        optimizer = torch.optim.adagrad
    elif name.lower() == 'amsgrad':
        from ..learn.optim import AMSGrad
        optimizer = AMSGrad
    else:
        raise SynthtorchError(f'Optimizer: "{name}" not a valid optimizer routine or not supported.')
    return optimizer


def get_loss(name:str):
    """ get a loss function by name """
    if name == 'mse' or name is None:
        loss = nn.MSELoss()
    elif name == 'cp':
        loss = CosineProximityLoss()
    elif name == 'mae':
        loss = nn.L1Loss()
    elif name == 'bce':
        loss = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Loss function {name} not supported.')
    return loss


def init_weights(net, init_type='kaiming', init_gain=0.02):
    """
    Initialize network weights
    (inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)

    Args:
        net (nn.Module): network to be initialized
        init_type (str): the name of an initialization method: normal, xavier, kaiming, or orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.

    Returns:
        None
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif (classname.find('BatchNorm') != -1 or
              classname.find('InstanceNorm') != -1 or
              classname.find('GroupNorm') != -1):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    if hasattr(net, 'n_seg'):  # handle segae last layer initialization
        if net.last_init is not None:
            initial_values = torch.tensor(net.last_init)
        else:
            initial_values = torch.from_numpy(np.sort(np.random.rand(net.n_seg) * 2))
        net.finish[2].weight.data = (initial_values.type_as(net.finish[2].weight.data)
                                                   .view(net.finish[2].weight.data.size()))

    if hasattr(net, 'all_conv'):  # handle ICNR initalization of upsample layers
        if net.all_conv and net.dim == 2:
            for m in net.upsampconvs: icnr(m[0].weight)


def icnr(m, scale=2, init=nn.init.kaiming_normal_):
    """ ICNR init of `x`, with `scale` and `init` function """
    ni,nf,h,w = m.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    m.data.copy_(k)
