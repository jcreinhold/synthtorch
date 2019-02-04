#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_exec

test the synthit command line interfaces for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 07, 2018
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

from synthnn.exec.nn_train import main as nn_train
from synthnn.exec.nn_predict import main as nn_predict
from synthnn.util.io import glob_nii, split_filename

try:
    import fastai
    from synthnn.exec.fa_train import main as fa_train
except ImportError:
    fastai = None


class TestCLI(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.nii_dir = os.path.join(wd, 'test_data', 'nii')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.tif_dir = os.path.join(wd, 'test_data', 'tif')
        self.out_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.out_dir, 'models'))
        self.train_dir = os.path.join(self.out_dir, 'train')
        os.mkdir(self.train_dir)
        os.mkdir(os.path.join(self.train_dir, '1'))
        os.mkdir(os.path.join(self.train_dir, '2'))
        nii = glob_nii(self.nii_dir)[0]
        tif = os.path.join(self.tif_dir, 'test.tif')
        path, base, ext = split_filename(nii)
        for i in range(8):
            shutil.copy(nii, os.path.join(self.train_dir, base + str(i) + ext))
            shutil.copy(tif, os.path.join(self.train_dir, '1', base + str(i) + '.tif'))
            shutil.copy(tif, os.path.join(self.train_dir, '2', base + str(i) + '.tif'))
        self.train_args = f'-s {self.train_dir} -t {self.train_dir}'.split()
        self.predict_args = f'-s {self.train_dir} -o {self.out_dir}/test'.split()
        self.jsonfn = f'{self.out_dir}/test.json'

    def __modify_ocf(self, jsonfn, multi=1):
        with open(jsonfn, 'r') as f:
            arg_dict = json.load(f)
        with open(jsonfn, 'w') as f:
            arg_dict['Required']['predict_dir'] = [f'{self.nii_dir}'] * multi
            arg_dict['Required']['predict_out'] = f'{self.out_dir}/test'
            json.dump(arg_dict, f, sort_keys=True, indent=2)

    def test_nconv_nopatch_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_nopatch.mdl -na nconv -ne 1 -nl 2 -ps 0 -bs 2 '
                                  f'--plot-loss {self.out_dir}/loss.png -ocf {self.jsonfn} '
                                  f'-vsd {self.train_dir} -vtd {self.train_dir}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_patch_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 ' 
                                  f'-ocf {self.jsonfn} -bs 2').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_lr_scheduler_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -lrs -v').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_data_aug_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_nopatch.mdl -na nconv -ne 1 -nl 2 -ps 0 -bs 2 '
                                  f'--plot-loss {self.out_dir}/loss.png -ocf {self.jsonfn} '
                                  f'-vsd {self.train_dir} -vtd {self.train_dir} -p 1 1 1 1 -r 10 -ts 0.5 -sc 0.1 '
                                  f'-hf -vf -g 0.1 -gn 0.2 -std 1 -tx -ty').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_clip_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -c 0.25').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 -bs 2 --net3d '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_vae_2d_3l_cli(self):
        train_args = f'-s {self.train_dir}/1/ -t {self.train_dir}/2/'.split()
        args = train_args + (f'-o {self.out_dir}/vae.mdl -na vae -ne 1 -nl 3 -cbp 2 -bs 4 --tiff '
                             f'--img-dim 256 256 --latent-size 10 -ocf {self.jsonfn} -sa 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        #TODO: cannot test 2d prediction here because nii needs to be same size as tiff, fix

    def test_vae_2d_5l_cli(self):
        train_args = f'-s {self.train_dir}/1/ -t {self.train_dir}/2/'.split()
        args = train_args + (f'-o {self.out_dir}/vae.mdl -na vae -ne 1 -nl 5 -cbp 1 -bs 4 --tiff '
                             f'--img-dim 256 256 --latent-size 10 -ocf {self.jsonfn} -sa 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        #TODO: cannot test 2d prediction here because nii needs to be same size as tiff, fix

    def test_vae_3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/vae.mdl -na vae -ne 1 -nl 3 -cbp 1 -ps 16 -bs 4 --net3d '
                                  f'--img-dim 16 16 16 --latent-size 10 -ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_no_skip_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 -bs 2 --net3d --no-skip '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_multimodal_cli(self):
        train_args = f'-s {self.train_dir}/1/ {self.train_dir}/1/ -t {self.train_dir}/2/ {self.train_dir}/2/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 -bs 2 --tiff '
                             f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_multimodal_cli(self):
        train_args = f'-s {self.train_dir} {self.train_dir} -t {self.train_dir} {self.train_dir}'.split()
        args = train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 ' 
                             f'-ocf {self.jsonfn} -bs 2').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_multimodal_tiff_cli(self):
        train_args = f'-s {self.train_dir}/1/ {self.train_dir}/1/ -t {self.train_dir}/2/ {self.train_dir}/2/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 '
                             f'-ocf {self.jsonfn} -bs 2 --tiff').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
