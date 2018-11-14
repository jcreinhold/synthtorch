#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.fa_train

command line interface to train a deep convolutional neural network for
synthesis of MR (brain) images with fastai

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

import argparse
import logging
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import fastai as fai
    import fastai.vision as faiv
    import torch
    from torch import nn
    from torchvision.transforms import Compose
    from niftidataset import NiftiDataset
    import niftidataset.transforms as nd_tfms
    from synthnn import split_filename
    from synthnn.models.unet import Unet
    from synthnn.util.io import AttrDict


def arg_parser():
    parser = argparse.ArgumentParser(description='train a CNN for MR image synthesis')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True,
                          help='path to directory with source images')
    required.add_argument('-t', '--target-dir', type=str, required=True,
                          help='path to directory with target images')

    options = parser.add_argument_group('Options')
    options.add_argument('-vs', '--valid-source-dir', type=str,
                          help='path to directory with source images for validation')
    options.add_argument('-vt', '--valid-target-dir', type=str,
                          help='path to directory with target images for validation')
    options.add_argument('-o', '--output', type=str, default=None,
                         help='path to output the trained model')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                         help='optional directory of brain masks for images')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    options.add_argument('-vc', '--validation-count', type=int, default=0,
                         help="number of datasets to use in validation")
    options.add_argument('-csv', '--out-csv', type=str, default='history',
                         help='name of output csv which holds training log')
    options.add_argument('-ocf', '--out-config-file', type=str, default='config.json',
                         help='output a config file for the options used in this experiment '
                              '(saves them as a json file with the name as input in this argument)')

    synth_options = parser.add_argument_group('Synthesis Options')
    synth_options.add_argument('-ps', '--patch-size', type=int, default=64,
                               help='patch size^3 extracted from image (0 for a full slice, '
                                    'sample-axis must be defined if full slice used) [Default=64]')

    nn_options = parser.add_argument_group('Neural Network Options')
    nn_options.add_argument('-n', '--n-jobs', type=int, default=None,
                            help='number of CPU processors to use for data loading [Default=None (all cpus)]')
    nn_options.add_argument('-ne', '--n-epochs', type=int, default=100,
                            help='number of epochs [Default=100]')
    nn_options.add_argument('-nl', '--n-layers', type=int, default=3,
                            help='number of layers to use in network (different meaning per arch) [Default=3]')
    nn_options.add_argument('-ks', '--kernel-size', type=int, default=3,
                            help='convolutional kernel size (cubed) [Default=3]')
    nn_options.add_argument('-sa', '--sample-axis', type=int, default=None,
                            help='axis on which to sample for 2d (None for random orientation) [Default=None]')
    nn_options.add_argument('-flr', '--flip-lr', action='store_true', default=False,
                            help='use flip lr data augmentation')
    nn_options.add_argument('-rot', '--rotate', action='store_true', default=False,
                            help='use rotation for data augmentation')
    nn_options.add_argument('-zm', '--zoom', action='store_true', default=False,
                            help='use zoom for data augmentation')
    nn_options.add_argument('-oc', '--one-cycle', action='store_true', default=False,
                            help='train using one-cycle policy (see "A Disciplined Approach...", Leslie Smith, 2018)')
    nn_options.add_argument('-in', '--include-neighbors', action='store_true', default=False,
                            help='take the nearest two slices when doing 2d sampling and append as new channels [Default=False]')
    nn_options.add_argument('-dp', '--dropout-prob', type=float, default=0,
                            help='dropout probability per conv block [Default=0]')
    nn_options.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                            help='learning rate of the neural network (uses Adam) [Default=1e-3]')
    nn_options.add_argument('-bs', '--batch-size', type=int, default=32,
                            help='batch size (num of images to process at once) [Default=5]')
    nn_options.add_argument('-cbp', '--channel-base-power', type=int, default=5,
                            help='batch size (num of images to process at once) [Default=5]')
    nn_options.add_argument('-pl', '--plot-loss', type=str, default=None,
                            help='plot the loss vs epoch and save at the filename provided here [Default=None]')
    nn_options.add_argument('--use-up-conv', action='store_true', default=False,
                            help='Use resize-convolution in the U-net as per the Distill article: '
                                 '"Deconvolution and Checkerboard Artifacts" [Default=False]')
    nn_options.add_argument('--add-two-up', action='store_true', default=False,
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
    nn_options.add_argument('-prl', '--preload', action='store_true', default=False,
                            help='preload dataset (memory intensive) vs loading data from disk each epoch')
    nn_options.add_argument('--disable-cuda', action='store_true', default=False,
                            help='Disable CUDA regardless of availability')
    return parser



def main(args=None):
    no_config_file = not sys.argv[1].endswith('.json') if args is None else not args[0].endswith('json')
    if no_config_file:
        args = arg_parser().parse_args(args)
    else:
        import json
        fn = sys.argv[1:][0] if args is None else args[0]
        with open(fn, 'r') as f:
            args = AttrDict(json.load(f))
    if args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(__name__)
    try:

        # get the desired neural network architecture
        model = Unet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                     channel_base_power=args.channel_base_power, add_two_up=args.add_two_up, normalization=args.normalization,
                     activation=args.activation, output_activation=args.out_activation, use_up_conv=args.use_up_conv, is_3d=False,
                     is_3_channel=args.include_neighbors)

        logger.debug(model)

        # put the model on the GPU if available and desired
        if torch.cuda.is_available() and not args.disable_cuda:
            model.cuda()

        # define device to put tensors on
        device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")

        # setup all transforms
        if args.patch_size > 0:
            nii_tfms = Compose([nd_tfms.RandomCrop2D(args.patch_size, args.sample_axis, include_neighbors=args.include_neighbors),
                                nd_tfms.ToTensor(),
                                nd_tfms.ToFastaiImage()])
        else:
             nii_tfms = Compose([nd_tfms.RandomSlice(args.sample_axis),
                                 nd_tfms.ToTensor(),
                                 nd_tfms.AddChannel(),
                                 nd_tfms.ToFastaiImage()])

        tds = NiftiDataset(args.source_dir, args.target_dir, nii_tfms, preload=args.preload)
        if args.valid_source_dir is not None and args.valid_target_dir is not None:
            vds = NiftiDataset(args.valid_source_dir, args.valid_target_dir, nii_tfms, preload=args.preload)
        else:
            vds = None

        tfms = []
        if args.flip_lr:
            tfms.append(faiv.flip_lr(p=0.5))
        if args.rotate:
            tfms.append(faiv.rotate(degrees=(-45, 45.), p=0.5))
        if args.zoom:
            tfms.append(faiv.zoom(scale=(0.95, 1.05), p=0.8))

        # define the fastai data class
        n_jobs = args.n_jobs if args.n_jobs is not None else fai.defaults.cpus
        idb = faiv.ImageDataBunch.create(tds, vds, bs=args.batch_size, ds_tfms=(tfms, []), num_workers=n_jobs,
                                         tfm_y=True, device=device)

        # setup the learner
        loss = nn.MSELoss()
        loss.__name__ = 'MSE'
        pth, base, _ = split_filename(args.output)
        learner = fai.Learner(idb, model, loss_func=loss, metrics=[loss], model_dir=pth)

        # enable fp16 (mixed) precision if desired
        if args.fp16:
            learner.to_fp16()

        # train the learner
        cb = fai.callbacks.CSVLogger(learner, args.out_csv)
        if not args.one_cycle:
            learner.fit(args.n_epochs, args.learning_rate, callbacks=cb)
        else:
            learner.fit_one_cycle(args.n_epochs, args.learning_rate, callbacks=[cb])

        # output a config file if desired
        if args.out_config_file is not None:
            import json
            import os
            arg_dict = vars(args)
            # add these keys so that the output config file can be edited for use in prediction
            arg_dict['trained_model'] = args.output + '.pth'
            arg_dict['predict_dir'] = None
            arg_dict['predict_out'] = None
            arg_dict['predict_mask_dir'] = None
            with open(args.out_config_file, 'w') as f:
                json.dump(arg_dict, f, sort_keys=True, indent=2)

        # save the trained model
        learner.save(args.output)

        # plot the loss vs epoch (if desired)
        if args.plot_loss is not None:
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            learner.recorder.plot_losses()
            plt.savefig(args.plot_loss)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
