#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.exec.nn_predict

command line interface to synthesize an MR (brain) image
from a trained neural network model (see synthnn.exec.nn_train)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import nibabel as nib
    import numpy as np
    import torch
    from synthnn import glob_nii, split_filename, SynthNNError
    from synthnn.util.io import AttrDict


def arg_parser():
    parser = argparse.ArgumentParser(description='predict an MR image from a trained neural net')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--predict-dir', type=str, required=True,
                          help='path to directory with source images on which to do prediction/synthesis')
    required.add_argument('-t', '--trained-model', type=str, required=True,
                          help='path to trained model')

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--predict-out', type=str, default=None,
                         help='path to output the synthesized image')
    options.add_argument('-m', '--predict-mask-dir', type=str, default=None,
                         help='optional directory of brain masks for images')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")

    nn_options = parser.add_argument_group('Neural Network Options')
    nn_options.add_argument('-n', '--n-jobs', type=int, default=0,
                            help='number of processors to use on CPU (use zero if CUDA enabled) [Default=0]')
    nn_options.add_argument('-bs', '--batch-size', type=int, default=5,
                              help='batch size (num of images to process at once) [Default=5]')
    nn_options.add_argument('--random-seed', default=0,
                              help='set random seed for reproducibility [Default=0]')
    nn_options.add_argument('--disable-cuda', action='store_true', default=False,
                            help='Disable CUDA regardless of availability')
    return parser


def main(args=None):
    no_config_file = args is not None or (args is None and len(sys.argv[1:]) > 1)
    if no_config_file:
        args = arg_parser().parse_args(args)
    else:
        import json
        with open(sys.argv[1:][0], 'r') as f:
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
        # set torch to use cuda if available (and desired) and set number of threads for the CPU (if enabled)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")
        torch.set_num_threads(args.n_jobs)

        # load the trained model
        if no_config_file:
            logger.warning('Loading entire serialized model in non-preferred way (without config file)')
            model = torch.load(args.trained_model)
        else:
            if args.nn_arch == 'nconv':
                from synthnn.models.nconvnet import Conv3dNLayerNet
                model = Conv3dNLayerNet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size)
            elif args.nn_arch == 'unet':
                from synthnn.models.unet import Unet
                model = Unet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                             channel_base_power=args.channel_base_power, add_two_up=args.add_two_up, normalization=args.normalization,
                             activation=args.activation, output_activation=args.out_activation, use_up_conv=args.use_up_conv)
            else:
                raise SynthNNError(f'Invalid NN type: {args.nn_arch}. {{nconv, unet}} are the only supported options.')
            model.load_state_dict(torch.load(args.trained_model, map_location=device))
        model.eval()
        logger.debug(model)

        # put the model on the GPU if available and desired
        if torch.cuda.is_available() and not args.disable_cuda:
            model.cuda()

        # set convenience variables and grab filenames of images to synthesize
        psz = model.patch_sz
        predict_dir = args.predict_dir if args.predict_dir is not None else args.source_dir
        output_dir = args.predict_out if args.predict_out is not None else os.getcwd() + '/syn_'
        predict_fns = glob_nii(predict_dir)
        for k, fn in enumerate(predict_fns):
            _, base, _ = split_filename(fn)
            logger.info(f'Starting synthesis of image: {base}. ({k+1}/{len(predict_fns)})')
            img_nib = nib.load(fn)
            img = img_nib.get_data().view(np.float32)  # set to float32 to save memory
            if psz > 0:
                out_img = np.zeros(img.shape)
                count_mtx = np.zeros(img.shape)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stride = psz // 2
                    indices = [torch.from_numpy(idxs) for idxs in np.indices(img.shape)]
                    for i in range(3):  # create blocks from imgs (and indices)
                        indices = [idxs.unfold(i, psz, stride) for idxs in indices]
                    x, y, z = [idxs.contiguous().view(-1, psz, psz, psz) for idxs in indices]
                dec_idxs = np.floor(np.percentile(np.arange(x.shape[0]), np.arange(0, 101, 5)))
                pct_complete = 0
                j = 0
                for i, (xx, yy, zz) in enumerate(zip(x, y, z)):
                    if i in dec_idxs:
                        logger.info(f'{pct_complete}% Complete')
                        pct_complete += 5
                    count_mtx[xx, yy, zz] = count_mtx[xx, yy, zz] + 1
                    if j == 0:
                        batch = np.zeros((args.batch_size,1,) + img[xx, yy, zz].shape, dtype=np.float32)
                        batch_idxs = [(xx,yy,zz)]
                        batch[j,0,...] = img[xx,yy,zz]
                        j += 1
                    elif j != args.batch_size:
                        batch_idxs.append((xx,yy,zz))
                        batch[j,0,...] = img[xx,yy,zz]
                        j += 1
                    else:
                        batch = torch.from_numpy(batch).to(device)
                        predicted = model.forward(batch).cpu().data.numpy()
                        for ii, (bx, by, bz) in enumerate(batch_idxs):
                            out_img[bx, by, bz] = out_img[bx, by, bz] + predicted[ii, 0, ...]
                        j = 0
                count_mtx[count_mtx == 0] = 1  # avoid division by zero
                out_img_nib = nib.Nifti1Image(out_img, img_nib.affine, img_nib.header)
            else:
                test_img_t = torch.from_numpy(img).to(device)[None, None, ...]
                out_img = np.squeeze(model.forward(test_img_t).cpu().data.numpy())
                out_img_nib = nib.Nifti1Image(out_img, img_nib.affine, img_nib.header)
            out_fn = output_dir + str(k) + '.nii.gz'
            out_img_nib.to_filename(out_fn)
            logger.info(f'Finished synthesis. Saved as: {out_fn}.')

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
