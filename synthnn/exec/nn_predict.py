#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.exec.nn_predict

command line interface to synthesize an MR (brain) image
with a trained pytorch NN (see nn_train)

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
    from synthnn.util.exec import get_args, get_device, setup_log
    from synthnn import glob_nii, split_filename, SynthNNError


######## Helper functions ########

def fwd(mdl, img): return mdl.forward(img).cpu().detach().numpy()


def batch2d(model, img, out_img, axis, device, bs, i, nsyn):
    s = np.transpose(img[:,i:i+bs,:,:],[1,0,2,3]) if axis == 0 else \
        np.transpose(img[:,:,i:i+bs,:],[2,0,1,3]) if axis == 1 else \
        np.transpose(img[:,:,:,i:i+bs],[3,0,1,2])
    img_b = torch.from_numpy(s).to(device)
    for _ in range(nsyn):
        if axis == 0:
            out_img[:,i:i+bs,:,:] = out_img[:,i:i+bs,:,:] + np.transpose(fwd(model, img_b), [1,0,2,3]) / nsyn
        elif axis == 1:
            out_img[:,:,i:i+bs,:] = out_img[:,:,i:i+bs,:] + np.transpose(fwd(model, img_b), [1,2,0,3]) / nsyn
        else:
            out_img[:,:,:,i:i+bs] = out_img[:,:,:,i:i+bs] + np.transpose(fwd(model, img_b), [1,2,3,0]) / nsyn


def save_imgs(out_img_nib, output_dir, k, logger):
    for i, oin in enumerate(out_img_nib):
        out_fn = output_dir + f'{k}_{i}.nii.gz'
        oin.to_filename(out_fn)
        logger.info(f'Finished synthesis. Saved as: {out_fn}.')


def get_overlapping_3d_idxs(psz, img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stride = psz // 2
        indices = [torch.from_numpy(idxs) for idxs in np.indices(img.shape[1:])]
        for i in range(3):  # create blocks from imgs (and indices)
            indices = [idxs.unfold(i, psz, stride) for idxs in indices]
        x, y, z = [idxs.contiguous().view(-1, psz, psz, psz) for idxs in indices]
    return x, y, z


######### Main routine ###########

def main(args=None):
    args, no_config_file = get_args(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    try:
        # define device to put tensors on
        device, use_cuda, n_gpus = get_device(args, logger)

        # determine if we enable dropout in prediction
        nsyn = args.monte_carlo or 1

        # load the trained model
        if args.nn_arch.lower() == 'nconv':
            from synthnn.models.nconvnet import SimpleConvNet
            model = SimpleConvNet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                                  n_input=args.n_input, n_output=args.n_output, is_3d=args.net3d)
        elif args.nn_arch.lower() == 'unet':
            from synthnn.models.unet import Unet
            model = Unet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                         channel_base_power=args.channel_base_power, add_two_up=args.add_two_up, normalization=args.normalization,
                         activation=args.activation, output_activation=args.out_activation, is_3d=args.net3d,
                         deconv=args.deconv, interp_mode=args.interp_mode, upsampconv=args.upsampconv, enable_dropout=nsyn > 1,
                         enable_bias=args.enable_bias, n_input=args.n_input, n_output=args.n_output)
        else:
            raise SynthNNError(f'Invalid NN type: {args.nn_arch}. {{nconv, unet}} are the only supported options.')
        state_dict = torch.load(args.trained_model, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.debug(model)

        # put the model on the GPU if available and desired
        if use_cuda: model.cuda(device=device)

        # setup and start prediction loop (whole slice by whole slice)
        axis = args.sample_axis or 0
        if axis < 0 or axis > 2 and not isinstance(axis,int):
            raise ValueError('sample_axis must be an integer between 0 and 2 inclusive')
        bs = args.batch_size // args.n_gpus if args.n_gpus > 1 and use_cuda else args.batch_size
        psz = args.patch_size
        predict_dir = args.predict_dir or args.valid_source_dir
        output_dir = args.predict_out or os.getcwd() + '/syn_'
        num_imgs = len(glob_nii(predict_dir[0]))
        if any([len(glob_nii(pd)) != num_imgs for pd in predict_dir]) or num_imgs == 0:
            raise SynthNNError('Number of images in prediction directories must be positive and have an equal number '
                               'of images in each directory (e.g., so that img_t1_1 aligns with img_t2_1 etc. for multimodal synth)')
        predict_fns = zip(*[glob_nii(pd) for pd in predict_dir])

        if args.net3d:  # 3D Synthesis Loop
            for k, fn in enumerate(predict_fns):
                _, base, _ = split_filename(fn[0])
                logger.info(f'Starting synthesis of image: {base}. ({k+1}/{num_imgs})')
                img_nib = nib.load(fn[0])
                img = np.stack([nib.load(f).get_data().view(np.float32) for f in fn])  # set to float32 to save memory
                if img.ndim == 3: img = img[np.newaxis, ...]
                if psz > 0:  # patch-based 3D synthesis
                    out_img = np.zeros((args.n_output,) + img.shape[1:])
                    count_mtx = np.zeros(img.shape[1:])
                    x, y, z = get_overlapping_3d_idxs(psz, img)
                    dec_idxs = np.floor(np.percentile(np.arange(x.shape[0]), np.arange(0, 101, 5)))
                    pct_complete = 0
                    j = 0
                    # The below for-loop handles collecting overlapping patches and putting
                    # them into a batch format that pytorch models expect (i.e., [N,C,H,W,D])
                    # and running the batch through the network (storing the results in out_img).
                    for i, (xx, yy, zz) in enumerate(zip(x, y, z)):
                        if i in dec_idxs:
                            logger.info(f'{pct_complete}% Complete')
                            pct_complete += 5
                        count_mtx[xx, yy, zz] = count_mtx[xx, yy, zz] + 1
                        if j == 0:
                            batch = np.zeros((args.batch_size,) + img[:, xx, yy, zz].shape, dtype=np.float32)
                            batch_idxs = [(xx, yy, zz)]
                            batch[j, ...] = img[:, xx, yy, zz]
                            j += 1
                        elif j != args.batch_size:
                            batch_idxs.append((xx, yy, zz))
                            batch[j, ...] = img[:, xx, yy, zz]
                            j += 1
                        else:
                            batch = torch.from_numpy(batch).to(device)
                            predicted = np.zeros(batch.shape)
                            for _ in range(nsyn):
                                predicted += fwd(model, batch)
                            for ii, (bx, by, bz) in enumerate(batch_idxs):
                                out_img[:, bx, by, bz] = out_img[:, bx, by, bz] + predicted[ii, ...]
                            j = 0
                    count_mtx[count_mtx == 0] = 1  # avoid division by zero
                    out_img_nib = [nib.Nifti1Image(out_img[i]/count_mtx, img_nib.affine, img_nib.header) for i in range(args.n_output)]
                else:  # whole-image-based 3D synthesis
                    test_img = torch.from_numpy(img).to(device)[None, ...]  # add empty batch dimension
                    out_img = np.squeeze(fwd(model, test_img))
                    out_img_nib = [nib.Nifti1Image(out_img[i], img_nib.affine, img_nib.header) for i in range(args.n_output)]
                save_imgs(out_img_nib, output_dir, k, logger)

        else:  # 2D Synthesis Loop -- goes by slice, does not use patches
            for k, fn in enumerate(predict_fns):
                _, base, _ = split_filename(fn[0])
                logger.info(f'Starting synthesis of image: {base}. ({k+1}/{num_imgs})')
                img_nib = nib.load(fn[0])
                img = np.stack([nib.load(f).get_data().view(np.float32) for f in fn])  # set to float32 to save memory
                if img.ndim == 3: img = img[np.newaxis, ...]
                out_img = np.zeros((args.n_output,) + img.shape[1:])
                num_batches = floor(img.shape[axis+1] / bs)  # add one to axis to ignore channel dim
                if img.shape[axis+1] / bs != num_batches:
                    lbi = int(num_batches * bs)  # last batch index
                    num_batches += 1
                    lbs = img.shape[axis+1] - lbi  # last batch size
                else:
                    lbi = None
                for i in range(num_batches if lbi is None else num_batches-1):
                    logger.info(f'Starting batch ({i+1}/{num_batches})')
                    batch2d(model, img, out_img, axis, device, bs, i*bs, nsyn)
                if lbi is not None:
                    logger.info(f'Starting batch ({num_batches}/{num_batches})')
                    batch2d(model, img, out_img, axis, device, lbs, lbi, nsyn)
                out_img_nib = [nib.Nifti1Image(out_img[i], img_nib.affine, img_nib.header) for i in range(args.n_output)]
                save_imgs(out_img_nib, output_dir, k, logger)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
