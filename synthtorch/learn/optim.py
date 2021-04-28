#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.learn.optim

define optimizer auxillary functions for neural network training

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 4, 2018
"""

__all__ = ['SGDW',
           'AMSGrad',
           'NesterovSGD',
           'NesterovSGDW']

import logging

import torch
from torch.optim import Optimizer, Adam, SGD

logger = logging.getLogger(__name__)


class SGDW(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, params, lr=0.01, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay)

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class AMSGrad(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(AMSGrad, self).__init__(params, lr, betas, eps, weight_decay, True)


class NesterovSGD(SGD):
    def __init__(self, params, lr, momentum=0.9, dampening=0, weight_decay=0):
        super().__init__(params, lr, momentum, dampening, weight_decay, True)


class NesterovSGDW(SGDW):
    def __init__(self, params, lr, momentum=0.9, dampening=0, weight_decay=0):
        super().__init__(params, lr, momentum, dampening, weight_decay, True)
