#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.util.config

create class for experiment configuration in the synthnn package

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 26, 2018
"""

__all__ = ['ExperimentConfig']

import json
import logging

from ..errors import SynthNNError

logger = logging.getLogger(__name__)


class ExperimentConfig(dict):

    def __init__(self, *args, **kwargs):
        self.predict_dir = None
        self.predict_out = None
        self.source_dir = None
        self.target_dir = None
        self.trained_model = None
        self.batch_size = None
        self.disable_cuda = None
        self.ext = None
        self.gpu_selector = None
        self.multi_gpu = None
        self.out_config_file = None
        self.patch_size = None
        self.pin_memory = None
        self.sample_axis = None
        self.seed = None
        self.verbosity = None
        self.activation = None
        self.add_two_up = None
        self.attention = None
        self.channel_base_power = None
        self.dropout_prob = None
        self.enable_bias = None
        self.init = None
        self.init_gain = None
        self.interp_mode = None
        self.kernel_size = None
        self.n_layers = None
        self.net3d = None
        self.nn_arch = None
        self.no_skip = None
        self.noise_lvl = None
        self.normalization = None
        self.ord_params = None
        self.out_activation = None
        self.separable = None
        self.checkpoint = None
        self.clip = None
        self.fp16 = None
        self.learning_rate = None
        self.loss = None
        self.lr_scheduler = None
        self.n_epochs = None
        self.n_jobs = None
        self.no_load_opt = None
        self.optimizer = None
        self.plot_loss = None
        self.valid_source_dir = None
        self.valid_split = None
        self.valid_target_dir = None
        self.weight_decay = None
        self.write_csv = None
        self.calc_var = None
        self.monte_carlo = None
        self.temperature_map = None
        self.img_dim = None
        self.latent_size = None
        self.freeze_last = None
        self.last_init = None
        self.n_seg = None
        self.ortho_penalty = None
        self.norm_penalty = None
        self.use_mse = None
        self.use_mask = None
        self.predict_seg = None
        self.initialize_seg = None
        self.seg_min = None
        self.n_gpus = None
        self.n_input = None
        self.n_output = None
        self.prob = None
        self.rotate = None
        self.translate = None
        self.scale = None
        self.hflip = None
        self.vflip = None
        self.gamma = None
        self.gain = None
        self.block = None
        self.noise_pwr = None
        self.mean = None
        self.std = None
        self.tfm_x = None
        self.tfm_y = None
        self.betas = None
        self.restart_period = None
        self.t_mult = None
        self.lrsd_weights = None
        super(ExperimentConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self._check_config()

    def _check_config(self):
        """ check to make sure requested configuration is valid """
        if self.ord_params is not None and self.n_output > 1:
            raise SynthNNError('Ordinal regression does not support multiple outputs.')

        if self.net3d and not (self.ext is None or 'nii' in self.ext):
            logger.warning(f'Cannot train a 3D network with {self.ext} images, creating a 2D network.')
            self.net3d = False

        if self.attention and self.net3d:
            logger.warning('Cannot use attention with 3D networks, not using attention.')
            self.attention = False

        if self.prob is not None:
            if (self.net3d or self.n_input > 1 or self.n_output > 1) and (self.prob[0] > 0 or self.prob[1] > 0):
                logger.warning('Cannot do affine, flipping or normalization data augmentation with multi-modal/3D networks.')
                self.prob[0], self.prob[1] = 0, 0
                self.rotate, self.translate, self.scale = 0, None, None
                self.hflip, self.vflip = False, False
                self.mean, self.std = None, None

        if (self.nn_arch.lower() != 'ordnet' and self.nn_arch.lower() != 'hotnet') and self.temperature_map:
            logger.warning('temperature_map is only a valid option when using OrdNet or HotNet.')
            self.temperature_map = False

        if self.loss == 'lrds' and not self.net3d:
            raise SynthNNError('low-rank and sparse decomposition is only supported for 3d')

        if self.loss == 'lrds' and len(self.target_dir) > 1:
            raise SynthNNError('low-rank and sparse decomposition is only supported for one output')

    @classmethod
    def load_json(cls, fn:str):
        """ handle loading from json file """
        with open(fn, 'r') as f:
            config = cls(_flatten(json.load(f)))  # dict comp. flattens first layer of dict
        return config

    @classmethod
    def from_argparse(cls, args):
        """ create an instance from a argument parser """
        args.n_gpus = 0
        args.n_input, args.n_output = len(args.source_dir), len(args.target_dir)
        arg_dict = _get_arg_dict(args)
        return cls(_flatten(arg_dict))

    def write_json(self, fn:str):
        """ write the experiment config to a file"""
        with open(fn, 'w') as f:
            arg_dict = _get_arg_dict(self.__dict__)
            json.dump(arg_dict, f, sort_keys=True, indent=2)


def _flatten(d): return {k: v for item in d.values() for k, v in item.items()}


def _get_arg_dict(args):
    arg_dict = {
        "Required": {
            "predict_dir": ["SET ME!"] if not hasattr(args,'predict_dir') else args.predict_dir,
            "predict_out": "SET ME!" if not hasattr(args,'predict_out') else args.predict_out,
            "source_dir": args.source_dir,
            "target_dir": args.target_dir,
            "trained_model": args.trained_model
        },
        "Options": {
            "batch_size": args.batch_size,
            "disable_cuda": args.disable_cuda,
            "ext": args.ext,
            "gpu_selector": args.gpu_selector,
            "multi_gpu": args.multi_gpu,
            "out_config_file": args.out_config_file,
            "patch_size": args.patch_size,
            "pin_memory": args.pin_memory,
            "sample_axis": args.sample_axis,
            "seed": args.seed,
            "verbosity": args.verbosity
        },
        "Optimizer Options": {
            "betas": args.betas,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "no_load_opt": args.no_load_opt,
            "optimizer": args.optimizer,
            "restart_period": args.restart_period,
            "t_mult": args.t_mult,
            "weight_decay": args.weight_decay
        },
        "Neural Network Options": {
            "activation": args.activation,
            "dropout_prob": args.dropout_prob,
            "init": args.init,
            "init_gain": args.init_gain,
            "kernel_size": args.kernel_size,
            "n_layers": args.n_layers,
            "net3d": args.net3d,
            "nn_arch": args.nn_arch,
        },
        "Training Options": {
            "checkpoint": args.checkpoint,
            "clip": args.clip,
            "fp16": args.fp16,
            "loss": args.loss,
            "n_epochs": args.n_epochs,
            "n_jobs": args.n_jobs,
            "plot_loss": args.plot_loss,
            "valid_source_dir": args.valid_source_dir,
            "valid_split": args.valid_split,
            "valid_target_dir": args.valid_target_dir,
            "write_csv": args.write_csv
        },
        "Prediction Options": {
            "calc_var": False if not hasattr(args,'calc_var') else args.calc_var,
            "monte_carlo": None if not hasattr(args,'monte_carlo') else args.monte_carlo,
        },
        "UNet Options": {
            "add_two_up": args.add_two_up,
            "attention": args.attention,
            "channel_base_power": args.channel_base_power,
            "enable_bias": args.enable_bias,
            "interp_mode": args.interp_mode,
            "no_skip": args.no_skip,
            "noise_lvl": args.noise_lvl,
            "normalization": args.normalization,
            "out_activation": args.out_activation,
            "separable": args.separable
        },
        "LRSDNet Options": {
            "lrsd_weights": args.lrsd_weights
        },
        "Ord/HotNet Options": {
            "ord_params": args.ord_params,
            "temperature_map": False if not hasattr(args,'temperature_map') else args.temperature_map
        },
        "VAE Options": {
            "img_dim": args.img_dim,
            "latent_size": args.latent_size if args.nn_arch == 'vae' else None
        },
        "SegAE Options": {
            "freeze_last": args.freeze_last,
            "initialize_seg": args.initialize_seg,
            "last_init": args.last_init,
            "n_seg": args.n_seg,
            "norm_penalty": args.norm_penalty,
            "ortho_penalty": args.ortho_penalty,
            "seg_min": args.seg_min,
            "use_mse": args.use_mse,
            "use_mask": args.use_mask,
            "predict_seg": False if not hasattr(args,'predict_seg') else args.predict_seg
        },
        "Internal": {
            "n_gpus": args.n_gpus,
            "n_input": args.n_input,
            "n_output": args.n_output
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
            "block": args.block,
            "noise_pwr": args.noise_pwr,
            "mean": args.mean,
            "std": args.std,
            "tfm_x": args.tfm_x,
            "tfm_y": args.tfm_y
        }
    }
    return arg_dict
