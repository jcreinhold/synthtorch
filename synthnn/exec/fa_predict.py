#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.fa_train

command line interface to synthesize an MR (brain) image
with a trained NN where the DNN was trained in fastai

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

import logging
from math import floor
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import nibabel as nib
    import numpy as np
    import torch
    from synthnn.models.unet import Unet
    from synthnn.util.io import AttrDict
    from synthnn import glob_nii, split_filename, SynthNNError


def fwd(mdl, img):
    out = mdl.forward(img).cpu().data.numpy()[:,0,:,:]
    return out


def batch(model, img, out_img, axis, device, bs, i):
    s = np.transpose(img[i:i+bs,:,:],[0,1,2])[:,np.newaxis,...] if axis == 0 else \
        np.transpose(img[:,i:i+bs,:],[1,0,2])[:,np.newaxis,...] if axis == 1 else \
        np.transpose(img[:,:,i:i+bs],[2,0,1])[:,np.newaxis,...]
    img_b = torch.from_numpy(s).to(device)
    if axis == 0:
        out_img[i:i+bs,:,:] = np.transpose(fwd(model, img_b), [0,1,2])
    elif axis == 1:
        out_img[:,i:i+bs,:] = np.transpose(fwd(model, img_b), [1,0,2])
    else:
        out_img[:,:,i:i+bs] = np.transpose(fwd(model, img_b), [1,2,0])


def main(args=None):
    no_config_file = not sys.argv[1].endswith('.json') if args is None else not args[0].endswith('json')
    if no_config_file:
        raise SynthNNError('Only configuration files are supported with fa-predict!')
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

        # load the model
        model = Unet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                     channel_base_power=args.channel_base_power, add_two_up=args.add_two_up, normalization=args.normalization,
                     activation=args.activation, output_activation=args.out_activation, use_up_conv=args.use_up_conv, is_3d=False,
                     is_3_channel=args.include_neighbors)
        model.load_state_dict(torch.load(args.trained_model))
        logger.debug(model)

        # put the model on the GPU if available and desired
        if torch.cuda.is_available() and not args.disable_cuda:
            model.cuda()

        # define device to put tensors on
        device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")

        axis = 0 if args.sample_axis is None else args.sample_axis
        bs = args.batch_size
        predict_fns = glob_nii(args.predict_dir)
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
                batch(model, img, out_img, axis, device, bs, i*bs)
            if lbi is not None:
                logger.info(f'Starting batch ({num_batches}/{num_batches})')
                batch(model, img, out_img, axis, device, lbs, lbi)
            out_img_nib = nib.Nifti1Image(out_img, img_nib.affine, img_nib.header)
            out_fn = args.predict_out + str(k) + '.nii.gz'
            out_img_nib.to_filename(out_fn)
            logger.info(f'Finished synthesis. Saved as: {out_fn}.')

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
