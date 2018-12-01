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
        for i in range(4):
            shutil.copy(nii, os.path.join(self.train_dir, base + str(i) + ext))
            shutil.copy(tif, os.path.join(self.train_dir, '1', base + str(i) + '.tif'))
            shutil.copy(tif, os.path.join(self.train_dir, '2', base + str(i) + '.tif'))
        self.train_args = f'-s {self.train_dir} -t {self.train_dir}'.split()
        self.predict_args = f'-s {self.train_dir} -o {self.out_dir}/test'.split()
        self.jsonfn = f'{self.out_dir}/test.json'

    def __modify_ocf(self, jsonfn):
        with open(jsonfn, 'r') as f:
            arg_dict = json.load(f)
        with open(jsonfn, 'w') as f:
            arg_dict['predict_dir'] = f'{self.nii_dir}'
            arg_dict['predict_out'] = f'{self.out_dir}/test'
            json.dump(arg_dict, f, sort_keys=True, indent=2)

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_fa(self):
        val_train_args = f'-vs 0.5'.split()
        args = self.train_args + val_train_args + (f'-o {self.out_dir}/fa -ne 2 -cbp 1 -nl 2 -bs 4 --plot-loss '
                                                   f'{self.out_dir}/loss.png -csv {self.out_dir}/history '
                                                   f'-ocf {self.jsonfn}').split()
        retval = fa_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_fa_tiff(self):
        val_train_args = f'-vs 0.5'.split()
        train_args = f'-s {self.train_dir}/1/ -t {self.train_dir}/2/'.split()
        args = train_args + val_train_args + (f'-o {self.out_dir}/fa -ne 2 -cbp 1 -nl 2 -bs 4 --plot-loss '
                                              f'{self.out_dir}/loss.png -csv {self.out_dir}/history '
                                              f'-ocf {self.jsonfn} --tiff').split()
        retval = fa_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_fa_whole_3d_vol(self):
        val_train_args = f'-vs 0.5'.split()
        args = self.train_args + val_train_args + (f'-o {self.out_dir}/fa --net3d -ps 0 -ne 1 -cbp 1 -nl 2 -bs 2 '
                                                   f'-csv {self.out_dir}/history -ocf {self.jsonfn}').split()
        retval = fa_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_fa_deconv(self):
        val_train_args = f'-vsd {self.train_dir} -vtd {self.train_dir} -vs 0'.split()
        args = self.train_args + val_train_args + (f'-o {self.out_dir}/fa -ne 1 -cbp 1 -nl 2 -bs 4 --plot-loss '
                                                   f'{self.out_dir}/loss.png -csv {self.out_dir}/history -ocf {self.jsonfn} '
                                                   f'-dc -bpe 1').split()
        retval = fa_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_fa_upsampconv(self):
        val_train_args = f'-vsd {self.train_dir} -vtd {self.train_dir} -vs 0'.split()
        args = self.train_args + val_train_args + (f'-o {self.out_dir}/fa -ne 1 -cbp 1 -nl 2 -bs 4 --plot-loss '
                                                   f'{self.out_dir}/loss.png -csv {self.out_dir}/history '
                                                   f'-ocf {self.jsonfn} -usc -bpe 1').split()
        retval = fa_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_fa_valid_dir(self):
        jsonfn = f'{self.out_dir}/test.json'
        val_train_args = f'-vsd {self.train_dir} -vtd {self.train_dir} -vs 0'.split()
        args = self.train_args + val_train_args + (f'-o {self.out_dir}/fa -ne 1 -cbp 1 -nl 2 -ps 0 -bs 4 '
                                                   f'-csv {self.out_dir}/history --one-cycle --disable-metrics '
                                                   f'-ocf {jsonfn} -bpe 1').split()
        retval = fa_train(args)
        self.assertEqual(retval, 0)

    def test_nn_nconv_nopatch_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_nopatch.mdl -na nconv -ne 1 -nl 2 -ps 0 -bs 2 '
                                  f'--plot-loss {self.out_dir}/loss.png -ocf {self.jsonfn} '
                                  f'-vsd {self.train_dir} -vtd {self.train_dir}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nn_nconv_patch_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 16 ' 
                                  f'-ocf {self.jsonfn} -bs 2').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self.__modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_nn_train_unet_cli(self):
        args = self.train_args + f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 1 -cbp 1 -ps 16 -bs 2'.split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
