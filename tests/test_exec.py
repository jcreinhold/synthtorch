#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_exec

test the synthtorch command line interfaces for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 07, 2018
"""

import os
import unittest

from synthtorch.exec.nn_train import main as nn_train
from synthtorch.exec.nn_predict import main as nn_predict
from ._test_funcs import TestCLI


class TestNConv(TestCLI):

    def test_nconv_nopatch_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_nopatch.mdl -na nconv -ne 1 -nl 2 -bs 2 -dm 3 '
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

    def test_nconv_preload_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 16 ' 
                                  f'-ocf {self.jsonfn} -bs 2 -pr').split()
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
                                  f'-ocf {self.jsonfn} -bs 2 -lrs cyclic -v -opt sgdw').split()
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
                                  f'--plot-loss {self.out_dir}/loss.png -ocf {self.jsonfn} -dm 3 '
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
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -dm 3 '
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

    def test_nconv_color_cli(self):
        train_args = f'-s {self.train_dir}/color/ -t {self.train_dir}/color/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 1 -nl 1 -cbp 1 -bs 2 -e png -co -dm 2 '
                             f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, color_out=True, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_nconv_color_tb_cli(self):
        train_args = f'-s {self.train_dir}/color/ -t {self.train_dir}/color/'.split()
        args = train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 1 -nl 1 -cbp 1 -bs 2 -e png -co -dm 2 '
                             f'-ocf {self.jsonfn} -tb').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, color_out=True, bs=1)
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
        args = self.train_args + (f'-o {self.out_dir}/nconv.mdl -na nconv -ne 1 -nl 1 -cbp 1 -bs 1 -dm 3 '
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
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_freeze_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -fr').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ic_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_sep3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
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
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 1 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -l cp').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_bce_cli(self):
        train_args = f'-s {self.train_dir} -t {self.train_dir}/mask/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 1 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                             f'-ocf {self.jsonfn} -l bce').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_mae_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 1 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -l mae').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_layernorm_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -nm layer').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_spectral_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -nm spectral').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_spectral_ks1_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -nm spectral -ks 1 1 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_semi3d1_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ic -s3 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_semi3d2_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ic -s3 2').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_acv_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ic -acv').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_ks331_ns_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -ks 3 3 1 -ns -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_weight_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -nm weight').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_selfattention_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 3 -bs 2 -e tif -ps 8 8 '
                             f'-ocf {self.jsonfn} -at self').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_cwattention_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 4 -bs 4 -e tif -ps 8 8 '
                             f'-ocf {self.jsonfn} -at channel').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_cwattention_3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 4 -bs 4 -ps 8 8 8 -dm 3 '
                                  f'-ocf {self.jsonfn} -at channel').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_cwattention_semi3d_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 2 -cbp 4 -bs 4 -ps 8 8 8 -dm 3 '
                                  f'-ocf {self.jsonfn} -at channel -ks 3 3 1 -ic -s3 2').split()
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
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 2 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
                                  f'-ocf {self.jsonfn} -nz 1').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unet_no_skip_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 --no-skip '
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
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
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

    def test_unet_color_2d_cli(self):
        train_args = f'-s {self.train_dir}/color/ -t {self.train_dir}/color/'.split()
        args = train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -bs 2 -e png -dm 2 '
                             f'-ocf {self.jsonfn} -co').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, color_out=True, bs=1)
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
        args = self.train_args + (f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 2 -dm 3 '
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
        args = self.train_args + (f'-o {self.out_dir}/vae.mdl -na vae -ne 1 -nl 3 -cbp 1 -ps 16 16 16 -bs 4 -dm 3 '
                                  f'--img-dim 16 16 16 --latent-size 10 -ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


if __name__ == '__main__':
    unittest.main()
