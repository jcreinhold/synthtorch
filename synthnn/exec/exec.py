#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.util.exec

helper functions for executables

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jan 28, 2018
"""

__all__ = ['determine_ext',
           'get_args',
           'setup_log']

import logging
import sys

from niftidataset import glob_imgs
from synthnn import SynthNNError, ExperimentConfig

logger = logging.getLogger(__name__)


def setup_log(verbosity):
    """ get logger with appropriate logging level and message """
    if verbosity == 1:
        level = logging.getLevelName('INFO')
    elif verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)


def get_args(args, arg_parser=None):
    """ handle arguments (through config file or argparser) for exec scripts """
    if arg_parser is not None:
        no_config_file = args is not None or (args is None and len(sys.argv[1:]) > 1) or sys.argv[1] == '-h' or sys.argv[1] == '--help'
    else:
        no_config_file = not sys.argv[1].endswith('.json') if args is None else not args[0].endswith('json')
    if no_config_file and arg_parser is None:
        raise SynthNNError('Only configuration files are supported with nn-predict! Create one with nn-train (see -ocf option).')
    elif no_config_file and arg_parser is not None:
        args = ExperimentConfig.from_argparse(arg_parser().parse_args(args))
    else:
        fn = sys.argv[1:][0] if args is None else args[0]
        args = ExperimentConfig.load_json(fn)
    return args, no_config_file


def determine_ext(d):
    """ given a directory determine if it contains supported images """
    exts = ('*.nii*', '*.tif*', '*.png')
    contains = [len(glob_imgs(d, ext)) > 0 for ext in exts]
    if sum(contains) == 0:
        raise SynthNNError(f'Directory {d} contains no supported images.')
    if sum(contains) > 1:
        raise SynthNNError(f'Directory {d} contains more than two types of supported images, '
                           f'remove unwanted images from directory')
    ext = [e for c, e in zip(contains, exts) if c][0]
    return ext
