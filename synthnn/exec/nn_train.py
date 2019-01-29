#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.exec.nn_train

command line interface to train a deep convolutional neural network for
synthesis of MR (brain) images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 28, 2018
"""

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import matplotlib
    matplotlib.use('agg')  # do not pull in GUI
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose
    from torch.utils.data.sampler import SubsetRandomSampler
    from niftidataset import MultimodalNiftiDataset, MultimodalTiffDataset
    import niftidataset.transforms as tfms
    from synthnn import SynthNNError
    from synthnn.util.exec import get_args, get_device, setup_log, write_out_config


def arg_parser():
    parser = argparse.ArgumentParser(description='train a CNN for MR image synthesis')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True, nargs='+',
                          help='path to directory with source images (multiple paths can be provided for multi-modal synthesis)')
    required.add_argument('-t', '--target-dir', type=str, required=True, nargs='+',
                          help='path to directory with target images (multiple paths can be provided for multi-modal synthesis)')

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--trained-model', type=str, default=None,
                         help='path to output the trained model')
    options.add_argument('-na', '--nn-arch', type=str, default='unet', choices=('unet', 'nconv'),
                         help='specify neural network architecture to use')
    options.add_argument('-vs', '--valid-split', type=float, default=0.2,
                          help='split the data in source_dir and target_dir into train/validation '
                               'with this split percentage [Default=0]')
    options.add_argument('-vsd', '--valid-source-dir', type=str, default=None, nargs='+',
                          help='path to directory with source images for validation, '
                               'see -vs for default action if this is not provided [Default=None]')
    options.add_argument('-vtd', '--valid-target-dir', type=str, default=None, nargs='+',
                          help='path to directory with target images for validation, '
                               'see -vs for default action if this is not provided [Default=None]')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    options.add_argument('-ocf', '--out-config-file', type=str, default=None,
                         help='output a config file for the options used in this experiment '
                              '(saves them as a json file with the name as input in this argument)')

    nn_options = parser.add_argument_group('Neural Network Options')
    nn_options.add_argument('-ps', '--patch-size', type=int, default=64,
                            help='patch size^3 extracted from image [Default=64]')
    nn_options.add_argument('-n', '--n-jobs', type=int, default=0,
                            help='number of CPU processors to use (use 0 if CUDA enabled) [Default=0]')
    nn_options.add_argument('-ne', '--n-epochs', type=int, default=100,
                            help='number of epochs [Default=100]')
    nn_options.add_argument('-nl', '--n-layers', type=int, default=3,
                            help='number of layers to use in network (different meaning per arch) [Default=3]')
    nn_options.add_argument('-ks', '--kernel-size', type=int, default=3,
                            help='convolutional kernel size (cubed) [Default=3]')
    nn_options.add_argument('-dc', '--deconv', action='store_true', default=False,
                            help='use transpose conv and strided conv for upsampling & downsampling respectively [Default=False]')
    nn_options.add_argument('-im', '--interp-mode', type=str, default='nearest', choices=('nearest','bilinear','trilinear'),
                            help='use this type of interpolation for upsampling (when deconv is false) [Default=nearest]')
    nn_options.add_argument('-dp', '--dropout-prob', type=float, default=0,
                            help='dropout probability per conv block [Default=0]')
    nn_options.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                            help='learning rate of the neural network (uses Adam) [Default=1e-3]')
    nn_options.add_argument('-bs', '--batch-size', type=int, default=5,
                            help='batch size (num of images to process at once) [Default=5]')
    nn_options.add_argument('-cbp', '--channel-base-power', type=int, default=5,
                            help='2 ** channel_base_power is the number of channels in the first layer '
                                 'and increases in each proceeding layer such that in the n-th layer there are '
                                 '2 ** (channel_base_power + n) channels [Default=5]')
    nn_options.add_argument('-pl', '--plot-loss', type=str, default=None,
                            help='plot the loss vs epoch and save at the filename provided here [Default=None]')
    nn_options.add_argument('-usc', '--upsampconv', action='store_true', default=False,
                            help='Use resize-convolution in the U-net as per the Distill article: '
                                 '"Deconvolution and Checkerboard Artifacts" [Default=False]')
    nn_options.add_argument('-atu', '--add-two-up', action='store_true', default=False,
                            help='Add two to the kernel size on the upsampling in the U-Net as '
                                 'per Zhao, et al. 2017 [Default=False]')
    nn_options.add_argument('-nm', '--normalization', type=str, default='instance', choices=('instance', 'batch', 'none'),
                            help='type of normalization layer to use in network [Default=instance]')
    nn_options.add_argument('-ac', '--activation', type=str, default='relu', choices=('relu', 'lrelu'),
                            help='type of activation to use throughout network except output [Default=relu]')
    nn_options.add_argument('-oac', '--out-activation', type=str, default='linear', choices=('relu', 'lrelu', 'linear'),
                            help='type of activation to use in network on output [Default=linear]')
    nn_options.add_argument('-mp', '--fp16', action='store_true', default=False,
                            help='enable mixed precision training')
    nn_options.add_argument('--disable-cuda', action='store_true', default=False,
                            help='Disable CUDA regardless of availability')
    nn_options.add_argument('-eb', '--enable-bias', action='store_true', default=False,
                            help='enable bias calculation in upsampconv layers and final conv layer [Default=False]')
    nn_options.add_argument('-sa', '--sample-axis', type=int, default=2,
                            help='axis on which to sample for 2d (None for random orientation when NIfTI images given) [Default=2]')
    nn_options.add_argument('--net3d', action='store_true', default=False, help='create a 3d network instead of 2d [Default=False]')
    nn_options.add_argument('--multi-gpu', action='store_true', default=False, help='use multiple gpus [Default=False]')
    nn_options.add_argument('--gpu-selector', type=int, nargs='+', default=None, help='use gpu(s) selected here, None '
                                                                                      'uses all available gpus if --multi-gpus enabled '
                                                                                      'else None uses first available GPU [Default=None]')
    nn_options.add_argument('--tiff', action='store_true', default=False, help='dataset are tiff images [Default=False]')
    nn_options.add_argument('--no-skip', action='store_true', default=False, help='do not use skip connections in unet [Default=False]')
    return parser


def main(args=None):
    args, no_config_file = get_args(args, arg_parser)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    try:
        # import and initialize mixed precision training package
        amp_handle = None
        if args.fp16:
            try:
                from apex import amp
                amp_handle = amp.init()
            except ImportError:
                logger.info('Mixed precision training (i.e., the package `apex`) not available.')

        use_3d = args.net3d and not args.tiff
        if args.net3d and args.tiff: logger.warning('Cannot train a 3D network with TIFF images, creating a 2D network.')
        n_input, n_output = len(args.source_dir), len(args.target_dir)

        # get the desired neural network architecture
        if args.nn_arch == 'nconv':
            from synthnn.models.nconvnet import SimpleConvNet
            logger.warning('The nconv network is for basic testing.')
            model = SimpleConvNet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                                  n_input=n_input, n_output=n_output, is_3d=use_3d)
        elif args.nn_arch == 'unet':
            from synthnn.models.unet import Unet
            model = Unet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                         channel_base_power=args.channel_base_power, add_two_up=args.add_two_up, normalization=args.normalization,
                         activation=args.activation, output_activation=args.out_activation, deconv=args.deconv, interp_mode=args.interp_mode,
                         upsampconv=args.upsampconv, enable_dropout=True, enable_bias=args.enable_bias, is_3d=use_3d,
                         n_input=n_input, n_output=n_output, no_skip=args.no_skip)
        else:
            raise SynthNNError(f'Invalid NN type: {args.nn_arch}. {{nconv, unet}} are the only supported options.')
        model.train(True)
        logger.debug(model)

        # define device to put tensors on
        device, use_cuda, n_gpus = get_device(args, logger)

        # put the model on the GPU if available and desired
        if use_cuda: model.cuda(device=device)
        use_multi = args.multi_gpu and n_gpus > 1 and use_cuda
        if args.multi_gpu and n_gpus <= 1: logger.warning('Multi-GPU functionality is not available on your system.')
        if use_multi:
            n_gpus = len(args.gpu_selector) if args.gpu_selector is not None else n_gpus
            logger.debug(f'Enabling use of {n_gpus} gpus')
            model = torch.nn.DataParallel(model, device_ids=args.gpu_selector)

        # check number of jobs requested and CPUs available
        num_cpus = os.cpu_count()
        if num_cpus < args.n_jobs:
            logger.warning(f'Requested more workers than available (n_jobs={args.n_jobs}, # cpus={num_cpus}). '
                           f'Setting n_jobs={num_cpus}.')
            args.n_jobs = num_cpus

        # control random cropping patch size (or if used at all)
        if not args.tiff:
            cropper = tfms.RandomCrop3D(args.patch_size) if args.net3d else tfms.RandomCrop2D(args.patch_size, args.sample_axis)
            tfm = [cropper] if args.patch_size > 0 else [] if args.net3d else [tfms.RandomSlice(args.sample_axis)]
        else:
            tfm = []
        tfm.append(tfms.ToTensor())

        # define dataset and split into training/validation set
        dataset = MultimodalNiftiDataset(args.source_dir, args.target_dir, Compose(tfm)) if not args.tiff else \
                  MultimodalTiffDataset(args.source_dir, args.target_dir, Compose(tfm))
        logger.debug(f'Number of training images: {len(dataset)}')

        if args.valid_source_dir is not None and args.valid_target_dir is not None:
            valid_dataset = MultimodalNiftiDataset(args.valid_source_dir, args.valid_target_dir, Compose(tfm)) if not args.tiff else \
                            MultimodalTiffDataset(args.valid_source_dir, args.valid_target_dir, Compose(tfm))
            logger.debug(f'Number of validation images: {len(valid_dataset)}')
            train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_jobs, shuffle=True)
            validation_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.n_jobs)
        else:
            # setup training and validation set
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(args.valid_split * num_train)
            validation_idx = np.random.choice(indices, size=split, replace=False)
            train_idx = list(set(indices) - set(validation_idx))

            train_sampler = SubsetRandomSampler(train_idx)
            validation_sampler = SubsetRandomSampler(validation_idx)

            # set up data loader for nifti images
            train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.n_jobs)
            validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=args.batch_size, num_workers=args.n_jobs)

        # train the model
        criterion = nn.MSELoss()
        logger.info(f'LR: {args.learning_rate:.5f}')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        use_valid = args.valid_split > 0 or (args.valid_source_dir is not None and args.valid_target_dir is not None)
        train_losses, validation_losses = [], []
        for t in range(args.n_epochs):
            # training
            t_losses = []
            if use_valid: model.train(True)
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)

                # Forward pass: Compute predicted y by passing x to the model
                tgt_pred = model(src)

                # Compute and store loss
                loss = criterion(tgt_pred, tgt)
                t_losses.append(loss.item())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                if args.fp16 and amp_handle is not None:
                    with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
            train_losses.append(t_losses)

            # validation
            v_losses = []
            if use_valid: model.train(False)
            with torch.set_grad_enabled(False):
                for src, tgt in validation_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    tgt_pred = model(src)
                    loss = criterion(tgt_pred, tgt)
                    v_losses.append(loss.item())
                validation_losses.append(v_losses)

            log = f'Epoch: {t+1} - Training Loss: {np.mean(t_losses):.2f}'
            if use_valid: log += f', Validation Loss: {np.mean(v_losses):.2f}'
            logger.info(log)

        # output a config file if desired
        if args.out_config_file is not None:
            write_out_config(args, n_gpus, n_input, n_output, use_3d)

        # save the trained model
        use_config_file = not no_config_file or args.out_config_file is not None
        if use_config_file:
            torch.save(model.state_dict(), args.trained_model)
        else:
            # save the whole model (if changes occur to pytorch, then this model will probably not be loadable)
            logger.warning('Saving the entire model. Preferred to create a config file and only save model weights')
            torch.save(model, args.trained_model)

        # strip multi-gpu specific attributes from saved model (so that it can be loaded easily)
        if use_multi and use_config_file:
            from collections import OrderedDict
            state_dict = torch.load(args.trained_model, map_location='cpu')
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            torch.save(new_state_dict, args.trained_model)

        # plot the loss vs epoch (if desired)
        if args.plot_loss is not None:
            plot_error = True if args.n_epochs <= 50 else False
            from synthnn import plot_loss
            if matplotlib.get_backend() != 'agg':
                import matplotlib.pyplot as plt
                plt.switch_backend('agg')
            ax = plot_loss(train_losses, ecolor='maroon', label='Train', plot_error=plot_error)
            _ = plot_loss(validation_losses, filename=args.plot_loss, ecolor='firebrick', ax=ax, label='Validation', plot_error=plot_error)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
