#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys
from glob import glob
import unittest

sys.path.append('../')
sys.path.append('../../')
from inputs.input_data_global_norm import read_htk

train_mean = None
train_std = None


class TestInputGlobalNorm(unittest.TestCase):

    def test(self):

        print('===== global norm input test =====')
        input_feature_path = '/n/sd8/inaguma/corpus/timit/fbank/'

        for data_type in ['train', 'dev', 'test']:
            print('=> ' + data_type)
            htk_paths = [join(input_feature_path, htk_dir)
                         for htk_dir in sorted(glob(join(input_feature_path,
                                                         data_type + '/*.htk')))]

            global train_mean
            global train_std
            if data_type == 'train':
                train_mean, train_std = read_htk(htk_paths=htk_paths,
                                                 normalize=True,
                                                 is_training=True)
            else:
                read_htk(htk_paths=htk_paths,
                         normalize=True,
                         is_training=False,
                         train_mean=train_mean,
                         train_std=train_std)


if __name__ == '__main__':
    unittest.main()
