#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.exec.nn_train

command line interface to train a deep convolutional neural network for
synthesis of MR (brain) images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 28, 2018
"""

import argparse
import logging
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import matplotlib
    matplotlib.use('agg')  # do not pull in GUI
    import numpy as np
    import torch
    from synthtorch import Learner
    from .exec import get_args, setup_log


######## Helper functions ########

def arg_parser():
    parser = argparse.ArgumentParser(description='train a CNN for MR image synthesis')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True, nargs='+',
                          help='path to directory with source images (multiple paths can be provided for multi-modal synthesis)')
    required.add_argument('-t', '--target-dir', type=str, required=True, nargs='+',
                          help='path to directory with target images (multiple paths can be provided for multi-modal synthesis)')
    required.add_argument('-o', '--trained-model', type=str, default=None,
                          help='path to output the trained model or (if model exists) continue training this model')

    options = parser.add_argument_group('Options')
    options.add_argument('-bs', '--batch-size', type=int, default=5,
                         help='batch size (num of images to process at once) [Default=5]')
    options.add_argument('-c', '--clip', type=float, default=None,
                         help='gradient clipping threshold [Default=None]')
    options.add_argument('-chk', '--checkpoint', type=int, default=None,
                         help='save the model every `checkpoint` epochs [Default=None]')
    options.add_argument('-csv', '--write-csv', type=str, default=None,
                         help="write the loss to a csv file of this filename [Default=None]")
    options.add_argument('--disable-cuda', action='store_true', default=False,
                         help='Disable CUDA regardless of availability')
    options.add_argument('-e', '--ext', type=str, default=None, choices=('nii','tif','png','jpg'),
                         help='extension of training/validation images [Default=None (.nii and .nii.gz)]')
    options.add_argument('-mp', '--fp16', action='store_true', default=False,
                         help='enable mixed precision training')
    options.add_argument('-l', '--loss', type=str, default=None, choices=('mse','mae','cp','bce'),
                         help='Use this specified loss function [Default=None, MSE for Unet]')
    options.add_argument('-mg', '--multi-gpu', action='store_true', default=False, help='use multiple gpus [Default=False]')
    options.add_argument('-n', '--n-jobs', type=int, default=0,
                            help='number of CPU processors to use (use 0 if CUDA enabled) [Default=0]')
    options.add_argument('-ocf', '--out-config-file', type=str, default=None,
                         help='output a config file for the options used in this experiment '
                              '(saves them as a json file with the name as input in this argument)')
    options.add_argument('-ps', '--patch-size', type=int, default=0,
                         help='patch size extracted from image [Default=0, i.e., whole slice or whole image]')
    options.add_argument('-pm','--pin-memory', action='store_true', default=False, help='pin memory in dataloader [Default=False]')
    options.add_argument('-pl', '--plot-loss', type=str, default=None,
                            help='plot the loss vs epoch and save at the filename provided here [Default=None]')
    options.add_argument('-sa', '--sample-axis', type=int, default=2,
                            help='axis on which to sample for 2d (None for random orientation when NIfTI images given) [Default=2]')
    options.add_argument('-sd', '--seed', type=int, default=0, help='set seed for reproducibility [Default=0]')
    options.add_argument('-vs', '--valid-split', type=float, default=0.2,
                          help='split the data in source_dir and target_dir into train/validation '
                               'with this split percentage [Default=0.2]')
    options.add_argument('-vsd', '--valid-source-dir', type=str, default=None, nargs='+',
                          help='path to directory with source images for validation, '
                               'see -vs for default action if this is not provided [Default=None]')
    options.add_argument('-vtd', '--valid-target-dir', type=str, default=None, nargs='+',
                          help='path to directory with target images for validation, '
                               'see -vs for default action if this is not provided [Default=None]')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")

    opt_options = parser.add_argument_group('Optimizer Options')
    opt_options.add_argument('-bt', '--betas', type=float, default=(0.9,0.99), nargs=2,
                             help='optimizer parameters (if using SGD, then the first element will be momentum '
                                  'and the second ignored) [Default=(0.9,0.99)]')
    opt_options.add_argument('-nlo', '--no-load-opt', action='store_true', default=False,
                             help='if loading a trained model, do not load the optimizer [Default=False]')
    opt_options.add_argument('-opt', '--optimizer', type=str, default='adam',
                             choices=('adam','adamw','sgd','sgdw','nesterov','adagrad','amsgrad','rmsprop'),
                             help='Use this optimizer to train the network [Default=adam]')
    opt_options.add_argument('-wd', '--weight-decay', type=float, default=0,
                             help="weight decay parameter for optimizer [Default=0]")

    sch_options = parser.add_argument_group('Scheduler Options')
    sch_options.add_argument('-cm', '--cycle-mode', type=str, default='triangluar', choices=('triangular','triangular2','exp_range'),
                             help='type of cycle for cyclic lr scheduler [Default=triangular]')
    sch_options.add_argument('-df', '--div-factor', type=float, default=25, help='divide LR by this amount for minimum LR [Default=25]')
    sch_options.add_argument('-lrs', '--lr-scheduler', type=str, default=None, choices=('cyclic', 'cosinerestarts'),
                             help='use a learning rate scheduler [Default=None]')
    sch_options.add_argument('-mr', '--momentum-range', type=float, nargs=2, default=(0.85,0.95),
                             help='range over which to inversely cycle momentum (does not work w/ all optimizers) [Default=(0.85,0.95)]')
    sch_options.add_argument('-nc', '--num-cycles', type=int, default=1,
                             help='number of cycles for cyclic learning rate scheduler [Default=1]')
    sch_options.add_argument('-rp', '--restart-period', type=int, default=None,
                             help='restart period for cosine annealing with restarts [Default=None]')
    sch_options.add_argument('-tm', '--t-mult', type=int, default=None,
                             help='multiplication factor for which the next restart period will extend or shrink '
                                  '(for cosine annealing with restarts) [Default=None]')

    nn_options = parser.add_argument_group('Neural Network Options')
    nn_options.add_argument('-ac', '--activation', type=str, default='relu',
                            choices=('relu', 'lrelu','prelu','elu','celu','selu','tanh','sigmoid'),
                            help='type of activation to use throughout network except output [Default=relu]')
    nn_options.add_argument('-dp', '--dropout-prob', type=float, default=0,
                            help='dropout probability per conv block [Default=0]')
    nn_options.add_argument('-eb', '--enable-bias', action='store_true', default=False,
                            help='enable bias calculation in upsampconv layers and final conv layer [Default=False]')
    nn_options.add_argument('-in', '--init', type=str, default='kaiming', choices=('normal', 'xavier', 'kaiming', 'orthogonal'),
                            help='use this type of initialization for the network [Default=kaiming]')
    nn_options.add_argument('-ing', '--init-gain', type=float, default=0.2,
                            help='use this initialization gain for initialization [Default=0.2]')
    nn_options.add_argument('-ks', '--kernel-size', type=int, default=3,
                            help='convolutional kernel size (squared or cubed) [Default=3]')
    nn_options.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                            help='learning rate for the optimizer [Default=1e-3]')
    nn_options.add_argument('-ne', '--n-epochs', type=int, default=100,
                            help='number of epochs [Default=100]')
    nn_options.add_argument('-nl', '--n-layers', type=int, default=3,
                            help='number of layers to use in network (different meaning per arch) [Default=3]')
    nn_options.add_argument('-3d', '--is-3d', action='store_true', default=False, help='create a 3d network instead of 2d [Default=False]')
    nn_options.add_argument('-na', '--nn-arch', type=str, default='unet',
                            choices=('unet','nconv','vae','segae','densenet','ordnet','lrsdnet','hotnet'),
                            help='specify neural network architecture to use')
    nn_options.add_argument('-nm', '--normalization', type=str, default='instance',
                            choices=('instance', 'batch', 'layer', 'weight', 'spectral', 'none'),
                            help='type of normalization layer to use in network [Default=instance]')
    nn_options.add_argument('-oac', '--out-activation', type=str, default='linear',
                            choices=('linear','relu', 'lrelu','prelu','elu','celu','selu','tanh','sigmoid'),
                            help='type of activation to use in network on output [Default=linear]')

    unet_options = parser.add_argument_group('UNet Options')
    unet_options.add_argument('-atu', '--add-two-up', action='store_true', default=False,
                              help='Add two to the kernel size on the upsampling in the U-Net as '
                                   'per Zhao, et al. 2017 [Default=False]')
    unet_options.add_argument('-acv', '--all-conv', action='store_true', default=False,
                              help='only use conv layers in unet (max pooling -> strided, upsamp -> shuffle) [Default=False]')
    unet_options.add_argument('-at', '--attention', action='store_true', default=False,
                              help='use attention gates in up conv layers in unet[Default=False]')
    unet_options.add_argument('-cbp', '--channel-base-power', type=int, default=5,
                              help='2 ** channel_base_power is the number of channels in the first layer '
                                   'and increases in each proceeding layer such that in the n-th layer there are '
                                   '2 ** (channel_base_power + n) channels [Default=5]')
    unet_options.add_argument('-ic', '--input-connect', action='store_true', default=False,
                              help='connect the input to the final layers via a concat skip connection [Default=False]')
    unet_options.add_argument('-im', '--interp-mode', type=str, default='nearest', choices=('nearest','bilinear','trilinear'),
                              help='use this type of interpolation for upsampling [Default=nearest]')
    unet_options.add_argument('-ns', '--no-skip', action='store_true', default=False,
                              help='do not use skip connections in unet [Default=False]')
    unet_options.add_argument('-nz', '--noise-lvl', type=float, default=0,
                              help='add this level of noise to model parameters [Default=0]')
    unet_options.add_argument('-rb', '--resblock', action='store_true', default=False,
                              help='use residual (addition) connections in unet blocks (all_conv must equal true to use) [Default=False]')
    unet_options.add_argument('-sp', '--separable', action='store_true', default=False,
                              help='use separable convolutions instead of full convolutions [Default=False]')
    unet_options.add_argument('-sx', '--softmax', action='store_true', default=False,
                              help='use softmax before last layer [Default=False]')

    lrsdnet_options = parser.add_argument_group('LRSDNet Options')
    lrsdnet_options.add_argument('-lrsd', '--lrsd-weights', type=float, nargs=2, default=None,
                                 help='penalties for lrsd [Default=None]')

    ordnet_options = parser.add_argument_group('OrdNet/HotNet Options')
    ordnet_options.add_argument('-cd', '--coord', action='store_true', default=False, help='use coords [Default=False]')
    ordnet_options.add_argument('-cx', '--cross', action='store_true', default=False, help='use cross connect [Default=False]')
    ordnet_options.add_argument('-ed', '--edge', action='store_true', default=False, help='use edge map [Default=False]')
    ordnet_options.add_argument('-lp', '--laplacian', action='store_true', default=False, help='use laplacian [Default=False]')
    ordnet_options.add_argument('-ord', '--ord-params', type=int, nargs=3, default=None,
                                help='ordinal regression params (start, stop, n_bins) [Default=None]')

    vae_options = parser.add_argument_group('VAE Options')
    vae_options.add_argument('-id', '--img-dim', type=int, nargs='+', default=None,
                             help='if using VAE, then input image dimension must be specified [Default=None]')
    vae_options.add_argument('-ls', '--latent-size', type=int, default=2048,
                             help='if using VAE, this controls latent dimension size [Default=2048]')

    segae_options = parser.add_argument_group('SegAE Options')
    segae_options.add_argument('-fl', '--freeze-last', action='store_true', default=False,
                               help='freeze the last layer for training [Default=False]')
    segae_options.add_argument('-is', '--initialize-seg', type=int, default=0,
                               help='number of epochs to initialize segmentation layers with seed [Default=0]')
    segae_options.add_argument('-nseg', '--n-seg', type=int, default=5, help='number of segmentation layers [Default=5]')
    segae_options.add_argument('-li', '--last-init', type=float, nargs='+', default=None,
                               help='initial numbers for last layer [Default=None]')
    segae_options.add_argument('-np', '--norm-penalty', type=float, default=1, help='weight for the norm penalty [Default=1]')
    segae_options.add_argument('-op', '--ortho-penalty', type=float, default=1, help='weight for the orthogonality penalty [Default=1]')
    segae_options.add_argument('-sm', '--seg-min', type=float, default=0, help='minimum prob in segmentation [Default=0]')
    segae_options.add_argument('-mse', '--use-mse', action='store_true', default=False,
                               help='use mse instead of cosine proximity in loss function [Default=False]')
    segae_options.add_argument('-mask', '--use-mask', action='store_true', default=False,
                               help='use brain mask on segmentation and output (during training and testing) [Default=False]')

    aug_options = parser.add_argument_group('Data Augmentation Options')
    aug_options.add_argument('-p', '--prob', type=float, nargs=5, default=None, help='probability of (Affine, Flip, Gamma, Block, Noise) [Default=None]')
    aug_options.add_argument('-r', '--rotate', type=float, default=0, help='max rotation angle [Default=0]')
    aug_options.add_argument('-ts', '--translate', type=float, default=0, help='max fractional translation [Default=0]')
    aug_options.add_argument('-sc', '--scale', type=float, default=0, help='max scale (1-scale,1+scale) [Default=0]')
    aug_options.add_argument('-hf', '--hflip', action='store_true', default=False, help='horizontal flip [Default=False]')
    aug_options.add_argument('-vf', '--vflip', action='store_true', default=False, help='vertical flip [Default=False]')
    aug_options.add_argument('-g', '--gamma', type=float, default=0, help='gamma (1-gamma,1+gamma) for (gain * x ** gamma) [Default=0]')
    aug_options.add_argument('-gn', '--gain', type=float, default=0, help='gain (1-gain,1+gain) for (gain * x ** gamma) [Default=0]')
    aug_options.add_argument('-blk', '--block', type=int, nargs=2, default=None, help='insert random blocks of this size range [Default=None]')
    aug_options.add_argument('-th', '--threshold', type=float, default=None, help='threshold for foreground for blocks, if none use mean [Default=None]')
    aug_options.add_argument('-pwr', '--noise-pwr', type=float, default=0, help='noise standard deviation/power [Default=0]')
    aug_options.add_argument('-mean', '--mean', type=float, nargs='+', default=None,
                             help='normalize input images with this mean (one entry per input directory) [Default=None]')
    aug_options.add_argument('-std', '--std', type=float, nargs='+', default=None,
                             help='normalize input images with this std (one entry per input directory) [Default=None]')
    aug_options.add_argument('-tx', '--tfm-x', action='store_true', default=True,
                             help='apply transforms to x (change this with config file) [Default=True]')
    aug_options.add_argument('-ty', '--tfm-y', action='store_true', default=False, help='apply transforms to y [Default=False]')
    return parser


######### Main routine ###########

def main(args=None):
    args, no_config_file = get_args(args, arg_parser)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    try:
        # set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        learner = Learner.train_setup(args)

        if args.fp16: learner.fp16()
        if args.multi_gpu: learner.multigpu()
        if args.lr_scheduler is not None: learner.lr_scheduler(**args)

        learner.fit(args.n_epochs, args.clip, args.checkpoint, args.trained_model)

        # output a config file if desired
        if args.out_config_file is not None: args.write_json(args.out_config_file)

        # save the trained model
        learner.save(args.trained_model, args.n_epochs)

        # plot/write the loss vs epoch (if desired)
        if args.plot_loss is not None: learner.record.plot_loss(args.plot_loss)
        if args.write_csv is not None: learner.record.write_csv(args.write_csv)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
