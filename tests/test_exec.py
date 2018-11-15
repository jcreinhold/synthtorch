#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_exec

test the synthit command line interfaces for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 07, 2018
"""

import os
import shutil
import tempfile
import unittest

from synthnn.exec.nn_train import main as nn_train
from synthnn.exec.nn_predict import main as nn_predict

try:
    import fastai
    from synthnn.exec.fa_train import main as fa_train
    from synthnn.exec.fa_predict import main as fa_predict
except ImportError:
    fastai = None


class TestCLI(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.out_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.out_dir, 'models'))
        self.train_args = f'-s {self.data_dir} -t {self.data_dir}'.split()
        self.predict_args = f'-s {self.data_dir} -o {self.out_dir}/test'.split()

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_fa(self):
        jsonfn = f'{self.out_dir}/test.json'
        val_train_args = f'-vs {self.data_dir} -vt {self.data_dir}'.split()
        args = self.train_args + val_train_args + (f'-o {self.out_dir}/fa -ne 2 -cbp 1 -nl 1 -ps 32 -bs 2 --plot-loss '
                                                   f'{self.out_dir}/loss.png -csv {self.out_dir}/history -ocf ' + jsonfn).split()
        retval = fa_train(args)
        self.assertEqual(retval, 0)
        import json
        with open(jsonfn, 'r') as f:
            arg_dict = json.load(f)
        with open(jsonfn, 'w') as f:
            arg_dict['predict_dir'] = f'{self.data_dir}'
            arg_dict['predict_out'] = f'{self.out_dir}/test'
            json.dump(arg_dict, f, sort_keys=True, indent=2)
        retval = fa_predict([jsonfn])
        self.assertEqual(retval, 0)

    def test_nn_nconv_nopatch_cli(self):
        args = self.train_args + f'-o {self.out_dir}/nconv_nopatch.mdl -na nconv -ne 2 -nl 1 -ps 0 --plot-loss {self.out_dir}/loss.png'.split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        args = self.predict_args + f'-t {self.out_dir}/nconv_nopatch.mdl'.split()
        retval = nn_predict(args)
        self.assertEqual(retval, 0)

    def test_nn_nconv_patch_cli(self):
        args = self.train_args + f'-o {self.out_dir}/nconv_patch.mdl -na nconv -ne 1 -nl 1 -ps 5'.split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)
        args = self.predict_args + f'-t {self.out_dir}/nconv_patch.mdl'.split()
        retval = nn_predict(args)
        self.assertEqual(retval, 0)

    def test_nn_train_unet_cli(self):
        args = self.train_args + f'-o {self.out_dir}/unet.mdl -na unet -ne 1 -nl 1 -cbp 1 -ps 16'.split()
        retval = nn_train(args)
        self.assertEqual(retval, 0)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
