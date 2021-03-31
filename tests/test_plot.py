#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_plot

test plotting functions for common runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 07, 2018
"""

import unittest

from synthtorch import plot_loss


class TestPlot(unittest.TestCase):

    def setUp(self):
        pass

    def test_nn_viz(self):
        all_losses = [[1, 2], [3, 4]]
        _ = plot_loss(all_losses)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
