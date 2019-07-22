#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.util.config

create class for experiment configuration in the synthtorch package

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 26, 2018
"""

__all__ = ['ExperimentConfig']

import json
import logging

from ..errors import SynthtorchError

logger = logging.getLogger(__name__)


class ExperimentConfig(dict):

    def __init__(self, *args, **kwargs):
        ### Setup all variables in the config class with default values
        # Required
        self.predict_dir        = None
        self.predict_out        = None
        self.source_dir         = None
        self.target_dir         = None
        self.trained_model      = None
        # Options
        self.batch_size         = None
        self.disable_cuda       = False
        self.ext                = None
        self.multi_gpu          = False
        self.out_config_file    = None
        self.patch_size         = 0
        self.pin_memory         = True
        self.sample_axis        = None
        self.seed               = 0
        self.verbosity          = 0
        # Optimizer Options
        self.betas              = (0.9, 0.99)
        self.learning_rate      = 0.001
        self.no_load_opt        = True
        self.optimizer          = "adam"
        self.weight_decay       = 0.01
        # Scheduler Options
        self.cycle_mode         = "triangular"
        self.div_factor         = 25
        self.lr_scheduler       = None
        self.momentum_range     = (0.85, 0.95)
        self.num_cycles         = 1
        self.pct_start          = 0.3
        self.restart_period     = None
        self.t_mult             = None
        # Neural Network Options
        self.activation         = "relu"
        self.dropout_prob       = 0
        self.init               = "kaiming"
        self.init_gain          = 0.02
        self.kernel_size        = 3
        self.n_layers           = 3
        self.is_3d              = True
        self.nn_arch            = "unet"
        # Training Options
        self.checkpoint         = None
        self.clip               = None
        self.fp16               = False
        self.loss               = None
        self.n_epochs           = 100
        self.n_jobs             = 6
        self.plot_loss          = None
        self.valid_source_dir   = None
        self.valid_split        = None
        self.valid_target_dir   = None
        self.write_csv          = None
        # Prediction Options
        self.calc_var           = False
        self.monte_carlo        = None
        # UNet Options
        self.all_conv           = False
        self.attention          = False
        self.channel_base_power = 4
        self.enable_bias        = True
        self.input_connect      = True
        self.interp_mode        = "nearest"
        self.no_skip            = False
        self.noise_lvl          = 0
        self.normalization      = "batch"
        self.out_activation     = "linear"
        self.resblock           = False
        self.semi_3d            = 0
        self.separable          = False
        self.softmax            = False
        # LRSDNet Options
        self.lrsd_weights       = None
        # Ord/HotNet Options
        self.beta               = 1.
        self.laplacian          = False
        self.ord_params         = None
        # VAE Options
        self.img_dim            = None
        self.latent_size        = None
        # SegAE Options
        self.freeze_last        = None
        self.initialize_seg     = 0
        self.last_init          = None
        self.n_seg              = None
        self.norm_penalty       = None
        self.ortho_penalty      = None
        self.predict_seg        = None
        self.use_mask           = None
        self.use_mse            = None
        # Internal
        self.n_gpus             = 1
        self.n_input            = 1
        self.n_output           = 1
        # Data Augmentation Options
        self.prob               = 0
        self.rotate             = 0
        self.translate          = 0
        self.scale              = 0
        self.hflip              = False
        self.vflip              = False
        self.gamma              = 0
        self.gain               = 0
        self.block              = None
        self.noise_pwr          = 0
        self.mean               = None
        self.std                = None
        self.threshold          = None
        self.tfm_x              = True
        self.tfm_y              = False
        # instantiate the class and check the setup
        super(ExperimentConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self._check_config()

    def _check_config(self):
        """ check to make sure requested configuration is valid """
        if self.ord_params is not None and self.n_output > 1:
            raise SynthtorchError('Ordinal regression does not support multiple outputs.')

        if self.is_3d and not (self.ext is None or 'nii' in self.ext):
            logger.warning(f'Cannot train a 3D network with {self.ext} images, creating a 2D network.')
            self.is_3d = False

        if self.attention and self.is_3d:
            logger.warning('Cannot use attention with 3D networks, not using attention.')
            self.attention = False

        if self.prob is not None:
            if (self.is_3d or self.n_input > 1 or self.n_output > 1) and (self.prob[0] > 0 or self.prob[1] > 0):
                logger.warning('Cannot do affine, flipping or normalization data augmentation with multi-modal/3D networks.')
                self.prob[0], self.prob[1] = 0, 0
                self.rotate, self.translate, self.scale = 0, None, None
                self.hflip, self.vflip = False, False

        if self.loss == 'lrds' and not self.is_3d:
            raise SynthtorchError('low-rank and sparse decomposition is only supported for 3d')

        if self.loss == 'lrds' and len(self.target_dir) > 1:
            raise SynthtorchError('low-rank and sparse decomposition is only supported for one output')

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
            "predict_dir": args.predict_dir if hasattr(args,'predict_dir') else ["SET ME!"],
            "predict_out": args.predict_out if hasattr(args,'predict_out') else "SET ME!",
            "source_dir": args.source_dir,
            "target_dir": args.target_dir,
            "trained_model": args.trained_model
        },
        "Options": {
            "batch_size": args.batch_size,
            "disable_cuda": args.disable_cuda,
            "ext": args.ext,
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
            "no_load_opt": args.no_load_opt,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay
        },
        "Scheduler Options": {
            "cycle_mode": args.cycle_mode,
            "div_factor": args.div_factor,
            "lr_scheduler": args.lr_scheduler,
            "momentum_range": args.momentum_range,
            "num_cycles": args.num_cycles,
            "pct_start": args.pct_start,
            "restart_period": args.restart_period,
            "t_mult": args.t_mult
        },
        "Neural Network Options": {
            "activation": args.activation,
            "dropout_prob": args.dropout_prob,
            "init": args.init,
            "init_gain": args.init_gain,
            "kernel_size": args.kernel_size,
            "n_layers": args.n_layers,
            "is_3d": args.is_3d,
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
            "calc_var": args.calc_var if hasattr(args,'calc_var') else False,
            "monte_carlo": args.monte_carlo if hasattr(args,'monte_carlo') else None,
        },
        "UNet Options": {
            "all_conv": args.all_conv,
            "attention": args.attention,
            "channel_base_power": args.channel_base_power,
            "enable_bias": args.enable_bias,
            "interp_mode": args.interp_mode,
            "input_connect": args.input_connect,
            "no_skip": args.no_skip,
            "noise_lvl": args.noise_lvl,
            "normalization": args.normalization,
            "out_activation": args.out_activation,
            "resblock": args.resblock,
            "semi_3d": args.semi_3d,
            "separable": args.separable,
            "softmax": args.softmax
        },
        "LRSDNet Options": {
            "lrsd_weights": args.lrsd_weights if hasattr(args,'lrsd_weights') else None
        },
        "Ord/HotNet Options": {
            "beta": args.beta if hasattr(args,'beta') else None,
            "laplacian": args.laplacian if hasattr(args,'laplacian') else None,
            "ord_params": args.ord_params if hasattr(args,'ord_params') else None
        },
        "VAE Options": {
            "img_dim": args.img_dim if hasattr(args,'img_dim') and args.nn_arch =='vae' else None,
            "latent_size": args.latent_size if hasattr(args,'latent_size') and args.nn_arch == 'vae' else None
        },
        "SegAE Options": {
            "freeze_last": args.freeze_last if hasattr(args,'freeze_last') else None,
            "initialize_seg": args.initialize_seg if hasattr(args,'initialize_seg') else 0,
            "last_init": args.last_init if hasattr(args,'last_init') else None,
            "n_seg": args.n_seg if hasattr(args,'n_seg') else None,
            "norm_penalty": args.norm_penalty if hasattr(args,'norm_penalty') else None,
            "ortho_penalty": args.ortho_penalty if hasattr(args,'ortho_penalty') else None,
            "use_mse": args.use_mse if hasattr(args,'use_mse') else None,
            "use_mask": args.use_mask if hasattr(args,'use_mask') else None,
            "predict_seg": args.predict_seg if hasattr(args,'predict_seg') else False
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
            "threshold": args.threshold,
            "tfm_x": args.tfm_x,
            "tfm_y": args.tfm_y
        }
    }
    return arg_dict
