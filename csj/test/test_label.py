#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from csj.path import Path
from csj.labels.transcript import read_sdb
from utils.measure_time_func import measure_time

path = Path(data_path='/n/sd8/inaguma/corpus/csj/data',
            config_path='../config')

label_paths = {
    'train_fullset': path.trans(data_type='train_fullset'),
    'train_subset': path.trans(data_type='train_subset'),
    'dev': path.trans(data_type='dev'),
    'eval1': path.trans(data_type='eval1'),
    'eval2': path.trans(data_type='eval2'),
    'eval3': path.trans(data_type='eval3')
}


class TestLabel(unittest.TestCase):

    def test(self):

        self.check()

    @measure_time
    def check(self):

        print('---------- train_subset ----------')
        read_sdb(label_paths=label_paths['train_subset'],
                 train_data_size='train_subset',
                 map_file_save_path='../config/mapping_files',
                 is_training=True,
                 save_map_file=True)

        print('---------- train_fullset ----------')
        read_sdb(label_paths=label_paths['train_fullset'],
                 train_data_size='train_fullset',
                 map_file_save_path='../config/mapping_files',
                 is_training=True,
                 save_map_file=True)

        for data_type in ['dev', 'eval1', 'eval2', 'eval3']:
            print('---------- %s ----------' % data_type)
            read_sdb(label_paths=label_paths[data_type],
                     train_data_size='train_subset',
                     map_file_save_path='../config/mapping_files',
                     is_test=True)


if __name__ == '__main__':
    unittest.main()
