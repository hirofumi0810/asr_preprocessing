#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import unittest

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data_global_norm import read_htk
from labels.ctc.phone import read_phone


class TestInputGlobalNorm(unittest.TestCase):
    def test(self):
        prep = Prepare()

        print('===== global norm =====')
        # train
        htk_dev_paths = [os.path.join(prep.data_root_path, htk_dir)
                         for htk_dir in sorted(glob.glob(os.path.join(prep.data_root_path, 'fbank/train/*.htk')))]
        train_mean, train_std = read_htk(htk_paths=htk_dev_paths,
                                         normalize=True,
                                         is_training=False)

        # dev
        htk_dev_paths = [os.path.join(prep.data_root_path, htk_dir)
                         for htk_dir in sorted(glob.glob(os.path.join(prep.data_root_path, 'fbank/dev/*.htk')))]
        read_htk(htk_paths=htk_dev_paths,
                 normalize=True,
                 is_training=False,
                 train_mean=train_mean,
                 train_std=train_std)

        # test
        htk_dev_paths = [os.path.join(prep.data_root_path, htk_dir)
                         for htk_dir in sorted(glob.glob(os.path.join(prep.data_root_path, 'fbank/test/*.htk')))]
        read_htk(htk_paths=htk_dev_paths,
                 normalize=True,
                 is_training=False,
                 train_mean=train_mean,
                 train_std=train_std)


if __name__ == '__main__':
    unittest.main()
