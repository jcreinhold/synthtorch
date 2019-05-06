#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.plot.loss

loss visualization plotting tools

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['plot_loss']

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    sns.set(style='white', font_scale=2)
except ImportError:
    logger.info('Seaborn not installed, so plots will not be as pretty. :-(')


def plot_loss(all_losses: List[list], figsize: Tuple[int,int]=(14,7), scale: int=0, ecolor: str='red',
              filename: Optional[str]=None, ax: Optional[object]=None, label: str='', plot_error: bool=True):
    """
    plot loss vs epoch for a given list (of lists) of loss values

    Args:
        all_losses (list): list of lists of loss values per epoch
        figsize (tuple): two ints in a tuple controlling figure size
        scale (int): two ints in a tuple controlling figure size
        ecolor (str): color of errorbars
        filename (str): if provided, save file at this path
        ax (matplotlib ax object): supply an ax if desired
        label (str): label for ax.plot
        plot_error (bool): plot error bars or nah

    Returns:
        ax (matplotlib ax object): ax that the plot was created on
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    avg_losses = np.array([np.mean(losses) for losses in all_losses]) * (10 ** scale)
    if plot_error:
        std_losses = np.array([np.std(losses) for losses in all_losses]) * (10 ** scale)
        ax.errorbar(np.arange(1,len(avg_losses)+1), avg_losses, yerr=std_losses, ecolor=ecolor, lw=3, fmt='none', alpha=0.5)
    ax.plot(np.arange(1,len(avg_losses)+1), avg_losses, lw=3, label=label)
    ax.set_title('Loss vs Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    if ax is not None:
        plt.legend()
    if filename is not None:
        plt.savefig(filename)
    return ax
