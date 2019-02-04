#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.util.optim

define optimizer auxillary functions for neural network training

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 4, 2018
"""

__all__ = ['BurnCosineLR']

import math

from torch.optim.lr_scheduler import _LRScheduler


class BurnCosineLR(_LRScheduler):
    """
    Set the learning rate of each parameter group using a cosine annealing
    schedule but starting at -T_max//3

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. [Default: 0.]
        last_epoch (int): The index of last epoch. [Default: -1.]
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.offset = -(T_max // 3)
        super(BurnCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch + self.offset) / self.T_max)) / 2
                for base_lr in self.base_lrs]
