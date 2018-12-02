#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.exec.nn_predict

command line interface to synthesize an MR (brain) image
with a trained pytorch NN (see fa_train or nn_train)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

import logging
from math import floor
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import nibabel as nib
    import numpy as np
    import torch
    from synthnn.util.io import AttrDict
    from synthnn import glob_nii, split_filename, SynthNNError


def fwd(mdl, img):
    out = mdl.forward(img).cpu().detach().numpy()[:,0,:,:]
    return out


def batch2d_proc(model, img, out_img, axis, device, bs, i, nsyn):
    s = np.transpose(img[i:i+bs,:,:],[0,1,2])[:,np.newaxis,...] if axis == 0 else \
        np.transpose(img[:,i:i+bs,:],[1,0,2])[:,np.newaxis,...] if axis == 1 else \
        np.transpose(img[:,:,i:i+bs],[2,0,1])[:,np.newaxis,...]
    img_b = torch.from_numpy(s).to(device)
    for _ in range(nsyn):
        if axis == 0:
            out_img[i:i+bs,:,:] = out_img[i:i+bs,:,:] + np.transpose(fwd(model, img_b), [0,1,2]) / nsyn
        elif axis == 1:
            out_img[:,i:i+bs,:] = out_img[:,i:i+bs,:] + np.transpose(fwd(model, img_b), [1,0,2]) / nsyn
        else:
            out_img[:,:,i:i+bs] = out_img[:,:,i:i+bs] + np.transpose(fwd(model, img_b), [1,2,0]) / nsyn


def main(args=None):
    no_config_file = not sys.argv[1].endswith('.json') if args is None else not args[0].endswith('json')
    if no_config_file:
        raise SynthNNError('Only configuration files are supported with nn-predict! Create one with fa-train or nn-train (-ocf).')
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
        # define device to put tensors on
        device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")

        # determine if we enable dropout in prediction
        nsyn = args.monte_carlo or 1

        # load the trained model
        if args.nn_arch.lower() == 'nconv':
            from synthnn.models.nconvnet import Conv3dNLayerNet
            model = Conv3dNLayerNet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size)
        elif args.nn_arch.lower() == 'unet':
            from synthnn.models.unet import Unet
            model = Unet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                         channel_base_power=args.channel_base_power, add_two_up=args.add_two_up, normalization=args.normalization,
                         activation=args.activation, output_activation=args.out_activation, is_3d=args.net3d, deconv=args.deconv,
                         interp_mode=args.interp_mode, upsampconv=args.upsampconv, enable_dropout=nsyn > 1, enable_bias=args.enable_bias)
        else:
            raise SynthNNError(f'Invalid NN type: {args.nn_arch}. {{nconv, unet}} are the only supported options.')
        state_dict = torch.load(args.trained_model, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict['model'])
        model.eval()

        logger.debug(model)

        # put the model on the GPU if available and desired
        if torch.cuda.is_available() and not args.disable_cuda:
            model.cuda()
            torch.backends.cudnn.benchmark = True

        # setup and start prediction loop (whole slice by whole slice)
        axis = 0 if args.sample_axis is None else args.sample_axis
        bs = args.batch_size
        psz = model.patch_sz
        predict_dir = args.predict_dir if args.predict_dir is not None else args.valid_source_dir
        output_dir = args.predict_out if args.predict_out is not None else os.getcwd() + '/syn_'
        predict_fns = glob_nii(predict_dir)

        if args.net3d: # 3D Synthesis Loop
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
                            batch = np.zeros((args.batch_size, 1,) + img[xx, yy, zz].shape, dtype=np.float32)
                            batch_idxs = [(xx, yy, zz)]
                            batch[j, 0, ...] = img[xx, yy, zz]
                            j += 1
                        elif j != args.batch_size:
                            batch_idxs.append((xx, yy, zz))
                            batch[j, 0, ...] = img[xx, yy, zz]
                            j += 1
                        else:
                            batch = torch.from_numpy(batch).to(device)
                            predicted = np.zeros(batch.shape)
                            for _ in range(nsyn):
                                predicted += model.forward(batch).cpu().detach().numpy()
                            for ii, (bx, by, bz) in enumerate(batch_idxs):
                                out_img[bx, by, bz] = out_img[bx, by, bz] + predicted[ii, 0, ...]
                            j = 0
                    count_mtx[count_mtx == 0] = 1  # avoid division by zero
                    out_img_nib = nib.Nifti1Image(out_img/count_mtx, img_nib.affine, img_nib.header)
                else:
                    test_img_t = torch.from_numpy(img).to(device)[None, None, ...]
                    out_img = np.squeeze(model.forward(test_img_t).cpu().detach().numpy())
                    out_img_nib = nib.Nifti1Image(out_img, img_nib.affine, img_nib.header)
                out_fn = output_dir + str(k) + '.nii.gz'
                out_img_nib.to_filename(out_fn)
                logger.info(f'Finished synthesis. Saved as: {out_fn}.')

        else:  # 2D Synthesis Loop
            for k, fn in enumerate(predict_fns):
                _, base, _ = split_filename(fn)
                logger.info(f'Starting synthesis of image: {base}. ({k+1}/{len(predict_fns)})')
                img_nib = nib.load(fn)
                img = img_nib.get_data().view(np.float32)  # set to float32 to save memory
                out_img = np.zeros(img.shape)
                num_batches = floor(img.shape[axis] / bs)
                if img.shape[axis] / bs != num_batches:
                    lbi = int(num_batches * bs) # last batch index
                    num_batches += 1
                    lbs = img.shape[axis] - lbi # last batch size
                else:
                    lbi = None
                for i in range(num_batches if lbi is None else num_batches-1):
                    logger.info(f'Starting batch ({i+1}/{num_batches})')
                    batch2d_proc(model, img, out_img, axis, device, bs, i*bs, nsyn)
                if lbi is not None:
                    logger.info(f'Starting batch ({num_batches}/{num_batches})')
                    batch2d_proc(model, img, out_img, axis, device, lbs, lbi, nsyn)
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
