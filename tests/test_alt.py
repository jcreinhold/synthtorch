#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_alt

test the synthtorch command line interfaces for runtime errors
with altdataset loaders (not available in this package)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 01, 2019
"""

import json
import os
import shutil
import tempfile
import unittest

from synthtorch.exec.nn_train import main as nn_train
from synthtorch.exec.nn_predict import main as nn_predict

try:
    import altdataset
except (ImportError, ModuleNotFoundError):
    raise unittest.SkipTest('Skipping test since annom toolbox not available.')


class TestCLI(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.csv_file = os.path.join(wd, 'test_data', 'csv', 'test.csv')
        self.out_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.out_dir, 'models'))
        self.train_args = f'-s {self.csv_file} -t {self.csv_file} -vsd {self.csv_file} -dm 1'.split()
        self.jsonfn = f'{self.out_dir}/test.json'

    def _modify_ocf(self, jsonfn, calc_var=False, mc=None):
        with open(jsonfn, 'r') as f:
            arg_dict = json.load(f)
        with open(jsonfn, 'w') as f:
            arg_dict['Required']['predict_dir'] = ''
            arg_dict['Required']['predict_out'] = f'{self.out_dir}/test'
            arg_dict['Prediction Options']['calc_var'] = calc_var
            arg_dict['Prediction Options']['monte_carlo'] = mc
            json.dump(arg_dict, f, sort_keys=True, indent=2)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


class Test1DNConv(TestCLI):

    def test_1d_nconv_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/nconv1d.mdl -na nconv -ne 1 -nl 2 -bs 2 '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)


class Test1DUnet(TestCLI):

    def test_1d_unet_cli(self):
        args = self.train_args + (f'-o {self.out_dir}/unet1d.mdl -na unet -ne 1 -nl 2 -bs 2 '
                                  f'-ocf {self.jsonfn}').split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)


if __name__ == '__main__':
    unittest.main()
