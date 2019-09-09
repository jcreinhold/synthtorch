#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_annom

test the synthtorch command line interfaces for runtime errors
with annom models (not available in this package)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 02, 2019
"""

import unittest

from synthtorch.exec.nn_train import main as nn_train
from synthtorch.exec.nn_predict import main as nn_predict
from ._test_funcs import TestCLI

try:
    import annom
except (ImportError, ModuleNotFoundError):
    raise unittest.SkipTest('Skipping test since annom toolbox not available.')


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


class TestLavaNet(TestCLI):

    def test_lava_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava_2d_softmax_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -sx -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava_2d_resblock_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -acv -rb -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava_2d_lap_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -l mae -ls 5 -l mae').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava_dropout_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.1 -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava_2d_freeze_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -fr -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava_3d_cli(self):
        args = self.train_args + (
            f'-o {self.out_dir}/lavanet.mdl -na lavanet -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestLava2Net(TestCLI):

    def test_lava2_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ {self.train_dir}/png/ -t {self.train_dir}/png/ {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava2_2d_softmax_cli(self):
        train_args = f'-s {self.train_dir}/png/ {self.train_dir}/png/ -t {self.train_dir}/png/ {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -sx -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava2_2d_resblock_cli(self):
        train_args = f'-s {self.train_dir}/png/ {self.train_dir}/png/ -t {self.train_dir}/png/ {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -acv -rb -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava2_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava2_2d_lap_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -l mae -ls 5 -l mae').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava2_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava2_dropout_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.1 -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava2_2d_freeze_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -fr -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_lava2_3d_cli(self):
        train_args = f'-s {self.train_dir} {self.train_dir} -t {self.train_dir} {self.train_dir}'.split()
        args = train_args + (
            f'-o {self.out_dir}/lava2net.mdl -na lava2net -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -ls 5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, multi=2)
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


class TestUnburnNet(TestCLI):

    def test_unburn_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/unburnnet.mdl -na unburnnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn_2d_softmax_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/unburnnet.mdl -na unburnnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -sx -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn_2d_resblock_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/unburnnet.mdl -na unburnnet -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -acv -rb -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unburnnet.mdl -na unburnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn_2d_lap_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unburnnet.mdl -na unburnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -l mae -ls 5 -l mae -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unburnnet.mdl -na unburnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn_2d_freeze_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unburnnet.mdl -na unburnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -fr -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn_3d_cli(self):
        args = self.train_args + (
            f'-o {self.out_dir}/unburnnet.mdl -na unburnnet -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


class TestUnburn2Net(TestCLI):

    def test_unburn2_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ {self.train_dir}/png/ -t {self.train_dir}/png/ {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/unburn2net.mdl -na unburn2net -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn2_2d_softmax_cli(self):
        train_args = f'-s {self.train_dir}/png/ {self.train_dir}/png/ -t {self.train_dir}/png/ {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/unburn2net.mdl -na unburn2net -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -sx -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn2_2d_resblock_cli(self):
        train_args = f'-s {self.train_dir}/png/ {self.train_dir}/png/ -t {self.train_dir}/png/ {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/unburn2net.mdl -na unburn2net -ne 1 -nl 2 -cbp 1 -ps 32 32 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -acv -rb -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn2_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unburn2net.mdl -na unburn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn2_2d_lap_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unburn2net.mdl -na unburn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -l mae -ls 5 -l mae -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn2_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unburn2net.mdl -na unburn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn2_2d_freeze_cli(self):
        train_args = f'-s {self.train_dir}/tif/ {self.train_dir}/tif/ -t {self.train_dir}/tif/ {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/unburn2net.mdl -na unburn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -fr -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_unburn2_3d_cli(self):
        train_args = f'-s {self.train_dir} {self.train_dir} -t {self.train_dir} {self.train_dir}'.split()
        args = train_args + (
            f'-o {self.out_dir}/unburn2net.mdl -na unburn2net -ne 1 -nl 1 -cbp 1 -ps 32 32 32 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -ls 5 -dp 0.5').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, multi=2)
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
                             f'-ocf {self.jsonfn} -ls 5 -mean 1 1 -std 1 1 --tfm-y -p 0 0 0 0 0').split()
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


class TestOCNet(TestCLI):

    def test_ocnet_2d_png_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 2 -cbp 1 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -ls 5 -id 64 64').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_2d_softmax_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 2 -cbp 1 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ic -sx -ls 5 -id 64 64').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_2d_resblock_cli(self):
        train_args = f'-s {self.train_dir}/png/ -t {self.train_dir}/png/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 2 -cbp 1 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -acv -rb -ls 5 -id 64 64').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_2d_noskip_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -ls 5 -id 64 64').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_2d_lap_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ns -l mae -ls 5 -l mae -id 64 64').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_2d_color_cli(self):
        train_args = f'-s {self.train_dir}/color/ -t {self.train_dir}/color/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -ls 5 -id 64 64 -co '
                             f'-p 1 1 1 0 1 -r 10 -ts 0.5 -sc 0.1 -mean 1 1 1 -std 1 1 1 '
                             f'-hf -vf -g 0.1 -gn 0.2 -pwr 1 -tx -ty').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, color_out=True, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -ls 5 -id 64 64').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_2d_sa_cli(self):
        train_args = f'-s {self.train_dir}/color/ -t {self.train_dir}/color/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -e png '
                             f'-ocf {self.jsonfn} -id 64 64 -at self -co').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, color_out=True, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_dropout_2d_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -dp 0.1 -ls 5 -id 64 64').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, mc=2, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_2d_freeze_cli(self):
        train_args = f'-s {self.train_dir}/tif/ -t {self.train_dir}/tif/'.split()
        args = train_args + (f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -e tif '
                             f'-ocf {self.jsonfn} -fr -id 64 64').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_3d_cli(self):
        args = self.train_args + (
            f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -id 32 32 32').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)

    def test_ocnet_3d_sa_cli(self):
        args = self.train_args + (
            f'-o {self.out_dir}/ocnet.mdl -na ocnet -ne 1 -nl 1 -cbp 1 -bs 4 -dm 3 '
            f'-ocf {self.jsonfn} -id 32 32 32 -at self').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        self._modify_ocf(self.jsonfn, bs=1)
        retval = nn_predict([self.jsonfn])
        self.assertEqual(retval, 0)


if __name__ == '__main__':
    unittest.main()
