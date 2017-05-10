#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import glob
import numpy as np

sys.path.append('../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data_global_norm import read_htk
from labels.ctc.phone import read_phone


class TestInputGlobalNorm(unittest.TestCase):
    def test(self):
        prep = Prepare()
        label_train_paths = prep.phone(data_type='train')
        label_dev_paths = prep.phone(data_type='dev')
        label_test_paths = prep.phone(data_type='test')


if __name__ == '__main__':
    unittest.main()
