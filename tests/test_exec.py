#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_exec

test the synthtorch command line interfaces for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 07, 2018
"""

import json
import os
import shutil
import tempfile
import unittest

from synthtorch.exec.nn_train import main as nn_train
from synthtorch.exec.nn_predict import main as nn_predict
from niftidataset import glob_imgs, split_filename

try:
    import annom
except ImportError:
    annom = None


class TestCLI(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.nii_dir = os.path.join(wd, 'test_data', 'nii')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.tif_dir = os.path.join(wd, 'test_data', 'tif')
        self.png_dir = os.path.join(wd, 'test_data', 'png')
        self.out_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.out_dir, 'models'))
        self.train_dir = os.path.join(self.out_dir, 'imgs')
        os.mkdir(self.train_dir)
        os.mkdir(os.path.join(self.train_dir, 'mask'))
        os.mkdir(os.path.join(self.train_dir, 'tif'))
        os.mkdir(os.path.join(self.train_dir, 'png'))
        nii = glob_imgs(self.nii_dir)[0]
        msk = glob_imgs(self.mask_dir)[0]
        tif = os.path.join(self.tif_dir, 'test.tif')
        png = os.path.join(self.png_dir, 'test.png')
        path, base, ext = split_filename(nii)
        for i in range(8):
            shutil.copy(nii, os.path.join(self.train_dir, base + str(i) + ext))
            shutil.copy(msk, os.path.join(self.train_dir, 'mask', base + str(i) + ext))
            shutil.copy(tif, os.path.join(self.train_dir, 'tif', base + str(i) + '.tif'))
            shutil.copy(png, os.path.join(self.train_dir, 'png', base + str(i) + '.png'))
        self.train_args = f'-s {self.train_dir} -t {self.train_dir}'.split()
        self.predict_args = f'-s {self.train_dir} -o {self.out_dir}/test'.split()
        self.jsonfn = f'{self.out_dir}/test.json'

    def _modify_ocf(self, jsonfn, multi=1, temperature_map=False, calc_var=False,
                    mc=None, predict_seg=False, png_out=False, tif_out=False):
        with open(jsonfn, 'r') as f:
            arg_dict = json.load(f)
        with open(jsonfn, 'w') as f:
            arg_dict['Required']['predict_dir'] = ([f'{self.nii_dir}'] * multi) if not png_out and not tif_out else \
                                                   [f'{self.train_dir}/png'] if png_out and not tif_out else \
                                                   [f'{self.train_dir}/tif']
            arg_dict['Required']['predict_out'] = f'{self.out_dir}/test'
            arg_dict['Prediction Options']['calc_var'] = calc_var
            arg_dict['Prediction Options']['monte_carlo'] = mc
            arg_dict['Prediction Options']['temperature_map'] = temperature_map
            arg_dict['SegAE Options']['predict_seg'] = predict_seg
            json.dump(arg_dict, f, sort_keys=True, indent=2)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


class TestNConv(TestCLI):

    def test_nconv_nopatch_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_nopatch.mdl -na nconv -ne 1 -nl 2 -bs 2 -3d '
                                  f'--plot-loss {self.out_dir}/loss.png -ocf {self.jsonfn} '
                                  f'-vsd {self.train_dir} -vtd {self.train_dir} -v').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_patch_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 16 ' 
                                  f'-ocf {self.jsonfn} -bs 2').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_swish_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 16 ' 
                                  f'-ocf {self.jsonfn} -bs 2 -ac swish').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_checkpoint_and_load_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 2 -nl 1 -ps 16 16 ' 
                                  f'-ocf {self.jsonfn} -bs 2 -chk 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        args = self.train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 2 -nl 1 -ps 16 16 ' 
                                  f'-ocf {self.jsonfn} -bs 2').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_cyclic_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -lrs cyclic -v -opt adamw').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_restarts_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -lrs cosinerestarts -tm 2 -rp 2 -v').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_amsgrad_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -v -opt amsgrad').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_nesterov_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -v -opt nsgd').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_nesterovw_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -v -opt nsgdw').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_sgdw_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -v -opt sgdw').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_adamw_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -v -opt adamw').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_weightdecay_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -v -wd 0.1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_writecsv_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 3 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -v -csv {self.out_dir}/test.csv').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)

    def test_nconv_data_aug_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv_nopatch.mdl -na nconv -ne 1 -nl 2 -bs 2 '
                             f'--plot-loss {self.out_dir}/loss.png -ocf {self.jsonfn} -e tif '
                             f'-p 1 1 1 1 1 -r 10 -ts 0.5 -sc 0.1 -mean 1 -std 1 '
                             f'-hf -vf -g 0.1 -gn 0.2 -pwr 1 -tx -ty -blk 5 6 -th 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_data_aug_3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_nopatch.mdl -na nconv -ne 1 -nl 2 -bs 2 '
                                  f'--plot-loss {self.out_dir}/loss.png -ocf {self.jsonfn} -3d '
                                  f'-vsd {self.train_dir} -vtd {self.train_dir} -p 0 0 1 1 1 '
                                  f'-g 0.01 -gn 0 -pwr 1 -tx -ty -blk 5 10 -mean 1 -std 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_clip_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 16 '
                                  f'-ocf {self.jsonfn} -bs 2 -c 0.25').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_whole_img_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -3d '
                                  f'-ocf {self.jsonfn} -bs 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_2d_crop_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 1 -nl 1 -cbp 1 -bs 2 -e tif -ps 8 8 '
                             f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_2d_var_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 1 -nl 1 -cbp 1 -bs 2 -e tif '
                             f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, calc_var=True)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 1 -nl 1 -cbp 1 -bs 2 -e png '
                             f'-ocf {self.jsonfn}  -p 1 1 0 0 0 ').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, calc_var=True, png_out=True)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_tif_predict_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 1 -nl 1 -cbp 1 -bs 2 -e tif '
                             f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, calc_var=True, tif_out=True)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_3d_var_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 1 -nl 1 -cbp 1 -bs 1 -3d '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, calc_var=True)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_multimodal_cli(self):
        train_args = f'-s {self.train_dir} {self.train_dir} -t {self.train_dir} {self.train_dir}'.split()
        args = train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 16 ' 
                             f'-ocf {self.jsonfn} -bs 2').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nconv_multimodal_tiff_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 16 '
                             f'-ocf {self.jsonfn} -bs 2 -e tif -th 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestDenseNet(TestCLI):

    def test_densenet_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/densenet.mdl -na densenet -ne 1 -bs 2 -e tif '
                             f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestUnet(TestCLI):

    def test_unet_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ic_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_sep3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -sp').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_sep2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -bs 2 -e tif '
                             f'-ocf {self.jsonfn} -sp').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_cp_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 1 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -l cp').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_bce_cli(self):
        train_args = f'-s {self.train_dir} -t {self.train_dir}/mask/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 1 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                             f'-ocf {self.jsonfn} -l bce').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_mae_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 1 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -l mae').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_layernorm_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -nm layer').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_spectral_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -nm spectral').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_spectral_ks1_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -nm spectral -ks 1 1 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_semi3d1_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ic -s3 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_semi3d2_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ic -s3 2').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_acv_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ic -acv').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_ns_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ns -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_weight_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -nm weight').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_attention_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 3 -bs 2 -e tif -ps 8 8 '
                             f'-ocf {self.jsonfn} -at').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_softmax_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 3 -bs 2 -e tif -ps 8 8 '
                             f'-ocf {self.jsonfn} -sx').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_noise_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 2 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -nz 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_no_skip_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -3d --no-skip '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_multimodal_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 -bs 2 -e tif '
                             f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_allconv_3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -acv').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_allconv_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -bs 2 -e tif -ps 16 16 '
                             f'-ocf {self.jsonfn} -acv').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_resblock_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -bs 2 -e tif -ps 16 16 '
                             f'-ocf {self.jsonfn} -acv -rb').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_resblock_2d_no_skip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -bs 2 -e tif -ps 16 16 '
                             f'-ocf {self.jsonfn} -acv -rb -ns').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_resblock_3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -3d '
                                  f'-ocf {self.jsonfn} -acv -rb').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestVAE(TestCLI):

    def test_vae_2d_3l_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/vae.mdl -na vae -ne 1 -nl 3 -cbp 2 -bs 4 -e tif -ps 32 32 '
                             f'--img-dim 32 32 --latent-size 10 -ocf {self.jsonfn} -sa 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        #TODO: cannot test 2d prediction here because nii needs to be same size as tiff, fix

    def test_vae_2d_5l_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/vae.mdl -na vae -ne 1 -nl 5 -cbp 1 -bs 4 -e tif -ps 32 32 '
                             f'--img-dim 32 32 --latent-size 10 -ocf {self.jsonfn} -sa 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        #TODO: cannot test 2d prediction here because nii needs to be same size as tiff, fix

    def test_vae_3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/vae.mdl -na vae -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 4 -3d '
                                  f'--img-dim 16 16 16 --latent-size 10 -ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestSegAE(TestCLI):

    def test_segae_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/segae.mdl -na segae -ne 2 -nl 3 -cbp 2 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -is 1 -nseg 4 -li 0.1 0.2 0.3 0.4 -fl').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_segae_2d_mse_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/segae.mdl -na segae -ne 1 -nl 3 -cbp 2 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} --use-mse -is 0 --clip 1 -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_segae_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/segae.mdl -na segae -ne 1 -nl 3 -cbp 2 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -is 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_segae_2d_mask_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/segae.mdl -na segae -ne 2 -nl 3 -cbp 2 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -mask -is 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_segae_2d_predict_seg_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/segae.mdl -na segae -ne 1 -nl 3 -cbp 1 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -is 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2, predict_seg=True)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_segae_3d_cli(self):
        train_args = f'-s {self.train_dir} {self.train_dir} -t {self.train_dir}'.split()
        args = train_args + (f'-o {self.out_dir}/segae.mdl -na segae -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -3d '
                             f'-ocf {self.jsonfn} -is 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_segae_3d_predict_seg_cli(self):
        train_args = f'-s {self.train_dir} {self.train_dir} -t {self.train_dir}'.split()
        args = train_args + (f'-o {self.out_dir}/segae.mdl -na segae -ne 1 -nl 1 -cbp 1 -bs 4 -3d '
                             f'-ocf {self.jsonfn} -is 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2, predict_seg=True)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestOrdNet(TestCLI):

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_ord_2d_cli(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
            valid = f'-vsd {self.train_dir}/tif/ -vtd {self.train_dir}/tif/'
            args = train_args + (f'-o {self.out_dir}/ordnet.mdl -na ordnet -ne 2 -nl 3 -cbp 1 -bs 4 -e tif '
                                 f'-ocf {self.jsonfn} -ord 1 10 10 {valid} -dp 0.5 -ic').split()
            retval = nn_train(args)
            self.assertEqual(retval, 0)
            self._modify_ocf(self.jsonfn, mc=2)
            retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_ord_2d_softmax_cli(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
            valid = f'-vsd {self.train_dir}/tif/ -vtd {self.train_dir}/tif/'
            args = train_args + (f'-o {self.out_dir}/ordnet.mdl -na ordnet -ne 2 -nl 3 -cbp 1 -bs 4 -e tif '
                                 f'-ocf {self.jsonfn} -ord 1 10 10 {valid} -dp 0.5 -ic -sx').split()
            retval = nn_train(args)
            self.assertEqual(retval, 0)
            self._modify_ocf(self.jsonfn, mc=2)
            retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_ord_3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/ordnet.mdl -na ordnet -ne 2 -nl 3 -cbp 1 -bs 4 -ps 16 16 16 -3d '
                                  f'-ocf {self.jsonfn} -ord 1 10 2 -vs 0.5 -ns -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_ord_2d_temperature_map_calc_var_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/ordnet.mdl -na ordnet -ne 2 -nl 3 -cbp 1 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ord 1 10 4 -vs 0.5 -ns -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, temperature_map=True, calc_var=True, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_ord_2d_temperature_map_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/ordnet.mdl -na ordnet -ne 2 -nl 3 -cbp 1 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ord 1 10 3 -vs 0.5 -ns -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, temperature_map=True, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_ord_2d_temperature_map_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/ordnet.mdl -na ordnet -ne 2 -nl 3 -cbp 1 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ord 1 10 3 -vs 0.5 -ns -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, temperature_map=True, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_ord_3d_temperature_map_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/ordnet.mdl -na ordnet -ne 2 -nl 3 -cbp 1 -bs 4 -ps 16 16 16 -3d '
                                  f'-ocf {self.jsonfn} -ord 1 10 5 -vs 0.5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, temperature_map=True, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestLRSDNet(TestCLI):

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_lrsd_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/lrsdnet.mdl -na lrsdnet -ne 1 -nl 2 -cbp 2 -ps 32 32 32 -bs 4 -3d '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestHotNet(TestCLI):

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -dp 0.5 -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_softmax_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -dp 0.5 -ic -sx').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_resblock_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -dp 0.5 -acv -rb').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_noskip_edge_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_lap_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -lp -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_mseonly_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -dp 0.5 -b1 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_maeonly_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -dp 0.5 -lp -b1 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_beta_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.5 -b1 10').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_2d_temp_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, temperature_map=True, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -3d '
                                  f'-ocf {self.jsonfn} -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(annom is None, 'Skipping test since annom toolbox not available.')
    def test_hot_3d_temp_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -3d '
                                  f'-ocf {self.jsonfn} -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, temperature_map=True, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


if __name__ == '__main__':
    unittest.main()
