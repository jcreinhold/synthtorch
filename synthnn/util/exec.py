#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.util.exec

helper functions for executables

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jan 28, 2018
"""

__all__ = ['get_args',
           'get_device',
           'setup_log',
           'write_out_config']

import json
import logging
import sys

import torch

from synthnn import SynthNNError


def setup_log(verbosity):
    if verbosity == 1:
        level = logging.getLevelName('INFO')
    elif verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)


def get_args(args, arg_parser=None):
    if arg_parser is not None:
        no_config_file = args is not None or (args is None and len(sys.argv[1:]) > 1) or sys.argv[1] == '-h' or sys.argv[1] == '--help'
    else:
        no_config_file = not sys.argv[1].endswith('.json') if args is None else not args[0].endswith('json')
    if no_config_file and arg_parser is None:
        raise SynthNNError('Only configuration files are supported with nn-predict! Create one with nn-train (see -ocf option).')
    elif no_config_file and arg_parser is not None:
        args = arg_parser().parse_args(args)
    else:
        fn = sys.argv[1:][0] if args is None else args[0]
        with open(fn, 'r') as f:
            args = AttrDict({k: v for item in json.load(f).values() for k, v in item.items()})  # dict comp. flattens first layer of dict
    return args, no_config_file


def get_device(args, logger):
    # define device to put tensors on
    cuda_avail = torch.cuda.is_available()
    use_cuda = cuda_avail and not args.disable_cuda
    if use_cuda: torch.backends.cudnn.benchmark = True
    if not cuda_avail and not args.disable_cuda: logger.warning('CUDA does not appear to be available on your system.')
    n_gpus = torch.cuda.device_count()
    if args.gpu_selector is not None:
        if len(args.gpu_selector) > n_gpus or any([gpu_id >= n_gpus for gpu_id in args.gpu_selector]):
            raise SynthNNError('Invalid number of gpus or invalid GPU ID input in --gpu-selector')
        cuda = f"cuda:{args.gpu_selector[0]}"  # arbitrarily choose first GPU given
    else:
        cuda = "cuda"
    device = torch.device(cuda if use_cuda else "cpu")
    return device, use_cuda, n_gpus


def write_out_config(args, n_gpus, n_input, n_output, use_3d):
    arg_dict = {
        "Required": {
            "predict_dir": ["SET ME!"],
            "predict_out": "SET ME!",
            "source_dir": args.source_dir,
            "target_dir": args.target_dir,
            "trained_model": args.trained_model
        },
        "Options": {
            "batch_size": args.batch_size,
            "disable_cuda": args.disable_cuda,
            "gpu_selector": args.gpu_selector,
            "multi_gpu": args.multi_gpu,
            "out_config_file": args.out_config_file,
            "patch_size": args.patch_size,
            "pin_memory": args.pin_memory,
            "sample_axis": args.sample_axis,
            "tiff": args.tiff,
            "verbosity": args.verbosity
        },
        "Neural Network Options": {
            "activation": args.activation,
            "add_two_up": args.add_two_up,
            "channel_base_power": args.channel_base_power,
            "dropout_prob": args.dropout_prob,
            "enable_bias": args.enable_bias,
            "init": args.init,
            "init_gain": args.init_gain,
            "interp_mode": args.interp_mode,
            "kernel_size": args.kernel_size,
            "n_layers": args.n_layers,
            "net3d": use_3d,
            "nn_arch": args.nn_arch,
            "no_skip": args.no_skip,
            "normalization": args.normalization,
            "out_activation": args.out_activation,
        },
        "Training Options": {
            "clip": args.clip,
            "fp16": args.fp16,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "n_epochs": args.n_epochs,
            "n_jobs": args.n_jobs,
            "plot_loss": args.plot_loss,
            "valid_source_dir": args.valid_source_dir,
            "valid_split": args.valid_split,
            "valid_target_dir": args.valid_target_dir
        },
        "Prediction Options": {
            "monte_carlo": None,
        },
        "VAE Options": {
            "img_dim": args.img_dim,
            "latent_size": args.latent_size if args.nn_arch == 'vae' else None
        },
        "Internal": {
            "n_gpus": n_gpus,
            "n_input": n_input,
            "n_output": n_output
        },
        "Data Augmentation Options": {
            "prob": args.prob,
            "rotate": args.rotate,
            "translate": args.translate,
            "scale": args.scale,
            "hflip": args.hflip,
            "vflip": args.vflip,
            "gamma": args.gamma,
            "gain": args.gain,
            "noise_std": args.noise_std,
            "tfm_x": args.tfm_x,
            "tfm_y": args.tfm_y
        }
    }
    with open(args.out_config_file, 'w') as f:
        json.dump(arg_dict, f, sort_keys=True, indent=2)


class AttrDict(dict):
    """
    make dictionary keys accessible via attributes
    used in nn_train and nn_predict to enable json config files
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
