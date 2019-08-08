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

from synthtorch.exec.nn_train import main as nn_train
from synthtorch.exec.nn_predict import main as nn_predict
from niftidataset import glob_imgs, split_filename

try:
    import annom
except (ImportError, ModuleNotFoundError):
    raise unittest.SkipTest('Skipping test since annom toolbox not available.')


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

    def _modify_ocf(self, jsonfn, multi=1, calc_var=False, mc=None, model=None,
                    predict_seg=False, png_out=False, tif_out=False):
        with open(jsonfn, 'r') as f:
            arg_dict = json.load(f)
        with open(jsonfn, 'w') as f:
            arg_dict['Required']['predict_dir'] = ([f'{self.nii_dir}'] * multi) if not png_out and not tif_out else \
                                                   [f'{self.train_dir}/png'] if png_out and not tif_out else \
                                                   [f'{self.train_dir}/tif']
            arg_dict['Required']['predict_out'] = f'{self.out_dir}/test'
            if model is not None: arg_dict['Neural Network Options']['nn_arch'] = model
            arg_dict['Prediction Options']['calc_var'] = calc_var
            arg_dict['Prediction Options']['monte_carlo'] = mc
            arg_dict['SegAE Options']['predict_seg'] = predict_seg
            json.dump(arg_dict, f, sort_keys=True, indent=2)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


class TestOrdNet(TestCLI):

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

    def test_ord_3d_cli(self):
        args = self.train_args + (
            f'-o {self.out_dir}/ordnet.mdl -na ordnet -ne 2 -nl 3 -cbp 1 -bs 4 -ps 16 16 16 -dm 3 '
            f'-ocf {self.jsonfn} -ord 1 10 2 -vs 0.5 -ns -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestLRSDNet(TestCLI):

    def test_lrsd_cli(self):
        args = self.train_args + (
            f'-o {self.out_dir}/lrsdnet.mdl -na lrsdnet -ne 1 -nl 2 -cbp 2 -ps 32 32 32 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestHotNet(TestCLI):

    def test_hot_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -dp 0.5 -ic').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_softmax_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -dp 0.5 -ic -sx').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_resblock_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -dp 0.5 -acv -rb').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_lap_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -l mae -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_mseonly_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -dp 0.5 -b1 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_maeonly_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -dp 0.5 -l mae -b1 0').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_beta_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.5 -b1 10').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_2d_freeze_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.5 -fr').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_hot_3d_cli(self):
        args = self.train_args + (
            f'-o {self.out_dir}/hotnet.mdl -na hotnet -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestBurnNet(TestCLI):

    def test_burn_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn_2d_softmax_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -sx -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn_2d_resblock_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -acv -rb -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn_2d_lap_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -l mae -ls 5 -l mae').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn_dropout_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.1 -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn_2d_freeze_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -fr -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn_3d_cli(self):
        args = self.train_args + (
            f'-o {self.out_dir}/burnnet.mdl -na burnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestBurn2Net(TestCLI):

    def test_burn2_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ {self.train_dir}/png/ -t {self.train_dir}/png/ {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -b1 1 -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn2_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn2_2d_mae_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -l mae -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn2_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn2_dropout_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.1 -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn2_2d_freeze_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -fr -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn2_3d_cli(self):
        train_args = f'-s {self.train_dir} {self.train_dir} -t {self.train_dir} {self.train_dir}'.split()
        args = train_args + (
            f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn2p12_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, model='burn2netp12')
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_burn2p21_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/burn2net.mdl -na burn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, model='burn2netp21')
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


if __name__ == '__main__':
    unittest.main()
