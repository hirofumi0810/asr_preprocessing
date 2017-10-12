#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test for transcript (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from librispeech.path import Path
from librispeech.transcript import read_trans
from utils.measure_time_func import measure_time

path = Path(
    data_path='/n/sd8/inaguma/corpus/librispeech/data')


class TestLabel(unittest.TestCase):

    def test(self):

        # frequency_threshold = 10
        self.check(train_data_size='train100h')
        self.check(train_data_size='train460h')
        self.check(train_data_size='train960h')

    @measure_time
    def check(self, train_data_size, frequency_threshold=10):

        print('==================================================')
        print('  train_data_size: %s' % str(train_data_size))
        print('  frequency_threshold: %s' % str(frequency_threshold))
        print('==================================================')

        print('---------- train ----------')
        read_trans(label_paths=path.trans(data_type=train_data_size),
                   train_data_size=train_data_size,
                   map_file_save_path='../config/mapping_files',
                   is_training=True,
                   frequency_threshold=frequency_threshold,
                   save_map_file=True,)

        for data_type in ['dev_clean', 'dev_other']:
            print('---------- %s ----------' % data_type)
            read_trans(label_paths=path.trans(data_type=data_type),
                       train_data_size=train_data_size,
                       map_file_save_path='../config/mapping_files',
                       frequency_threshold=frequency_threshold)

        for data_type in ['test_clean', 'test_other']:
            print('---------- %s ----------' % data_type)
            read_trans(label_paths=path.trans(data_type=data_type),
                       train_data_size=train_data_size,
                       map_file_save_path='../config/mapping_files',
                       is_test=True,
                       frequency_threshold=frequency_threshold)


if __name__ == '__main__':
    unittest.main()
