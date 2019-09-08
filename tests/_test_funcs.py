#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_annom

test the synthtorch command line interfaces for runtime errors
with annom models (not available in this package)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 02, 2019
"""

import json
import os
import shutil
import tempfile
import unittest

from niftidataset import glob_imgs, split_filename


class TestCLI(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.nii_dir = os.path.join(wd, 'test_data', 'nii')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.tif_dir = os.path.join(wd, 'test_data', 'tif')
        self.png_dir = os.path.join(wd, 'test_data', 'png')
        self.color_dir = os.path.join(wd, 'test_data', 'color')
        self.out_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.out_dir, 'models'))
        self.train_dir = os.path.join(self.out_dir, 'imgs')
        os.mkdir(self.train_dir)
        os.mkdir(os.path.join(self.train_dir, 'mask'))
        os.mkdir(os.path.join(self.train_dir, 'tif'))
        os.mkdir(os.path.join(self.train_dir, 'png'))
        os.mkdir(os.path.join(self.train_dir, 'color'))
        nii = glob_imgs(self.nii_dir)[0]
        msk = glob_imgs(self.mask_dir)[0]
        tif = os.path.join(self.tif_dir, 'test.tif')
        png = os.path.join(self.png_dir, 'test.png')
        color = os.path.join(self.color_dir, 'test.png')
        path, base, ext = split_filename(nii)
        for i in range(8):
            shutil.copy(nii, os.path.join(self.train_dir, base + str(i) + ext))
            shutil.copy(msk, os.path.join(self.train_dir, 'mask', base + str(i) + ext))
            shutil.copy(tif, os.path.join(self.train_dir, 'tif', base + str(i) + '.tif'))
            shutil.copy(png, os.path.join(self.train_dir, 'png', base + str(i) + '.png'))
            shutil.copy(color, os.path.join(self.train_dir, 'color', base + str(i) + '.png'))
        self.train_args = f'-s {self.train_dir} -t {self.train_dir}'.split()
        self.predict_args = f'-s {self.train_dir} -o {self.out_dir}/test'.split()
        self.jsonfn = f'{self.out_dir}/test.json'

    def _modify_ocf(self, jsonfn, multi=1, calc_var=False, mc=None, predict_seg=False,
                    png_out=False, tif_out=False, color_out=False, model=None, bs=None):
        with open(jsonfn, 'r') as f:
            arg_dict = json.load(f)
        with open(jsonfn, 'w') as f:
            use_nii = not png_out and not tif_out and not color_out
            arg_dict['Required']['predict_dir'] = ([f'{self.nii_dir}'] * multi) if use_nii else \
                                                   [f'{self.train_dir}/png'] if png_out else \
                                                   [f'{self.train_dir}/color'] if color_out else \
                                                   [f'{self.train_dir}/tif']
            arg_dict['Required']['predict_out'] = f'{self.out_dir}/test'
            arg_dict['Prediction Options']['calc_var'] = calc_var
            arg_dict['Prediction Options']['monte_carlo'] = mc
            arg_dict['SegAE Options']['predict_seg'] = predict_seg
            if bs is not None: arg_dict['Options']['batch_size'] = bs
            if model is not None: arg_dict['Neural Network Options']['nn_arch'] = model
            json.dump(arg_dict, f, sort_keys=True, indent=2)

    def tearDown(self):
        shutil.rmtree(self.out_dir)
