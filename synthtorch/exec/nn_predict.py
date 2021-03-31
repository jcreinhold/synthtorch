#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.exec.nn_predict

command line interface to synthesize an MR (brain) image
with a trained pytorch NN (see nn_train)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

import logging
import os
import random
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import numpy as np
    import torch
    from niftidataset import glob_imgs, split_filename
    from synthtorch import Learner, SynthtorchError
    from .exec import get_args, setup_log, determine_ext


######### Main routine ###########

def main(args=None):
    args, no_config_file = get_args(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    try:
        # set random seeds for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        # since prediction only uses one gpu (at most), make the batch size small enough to fit
        if args.n_gpus > 1: args.batch_size = args.batch_size // args.n_gpus

        learner = Learner.predict_setup(args)

        # determine how many samples we will use in prediction
        nsyn = (args.monte_carlo or 1) if (args.nn_arch not in ('hotnet', 'unburnnet', 'ocnet1', 'ocnet2')) else 1

        # get relevant prediction directories and determine extension
        predict_dir = args.predict_dir or args.valid_source_dir
        output_dir = args.predict_out or os.getcwd() + '/syn_'
        ext = determine_ext(predict_dir[0])

        # setup and start prediction loop
        axis = args.sample_axis or 0
        if axis < 0 or axis > 2 and not isinstance(axis, int):
            raise ValueError('sample_axis must be an integer between 0 and 2 inclusive')
        n_imgs = len(glob_imgs(predict_dir[0], ext))
        if n_imgs == 0: raise SynthtorchError('Prediction directory does not contain valid images.')
        if any([len(glob_imgs(pd, ext)) != n_imgs for pd in predict_dir]):
            raise SynthtorchError(
                'Number of images in prediction directories must have an equal number of images in each '
                'directory (e.g., so that img_t1_1 aligns with img_t2_1 etc. for multimodal synth)')
        predict_fns = zip(*[glob_imgs(pd, ext) for pd in predict_dir])

        if args.dim == 3 and args.patch_size is not None and args.calc_var:
            raise SynthtorchError('Patch-based 3D variance calculation not currently supported.')

        for k, fn in enumerate(predict_fns):
            _, base, ext = split_filename(fn[0])
            if 'png' in ext: ext = '.tif'  # force tif output in this case
            if 'nii' in ext: ext = '.nii.gz'  # force compressed output in this case
            logger.info(f'Starting synthesis of image: {base} ({k + 1}/{n_imgs})')
            out_imgs = learner.predict(fn, nsyn, args.calc_var)
            for i, oin in enumerate(out_imgs):
                out_fn = output_dir + f'{k}_{i}{ext}'
                if hasattr(oin, 'to_filename'):
                    oin.to_filename(out_fn)
                else:
                    oin.save(out_fn)
                logger.info(f'Finished synthesis. Saved as: {out_fn}')

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
