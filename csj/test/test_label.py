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
from utils.util import mkdir_join

path = Path(data_path='/n/sd8/inaguma/corpus/csj/data',
            config_path='../config')


class TestLabel(unittest.TestCase):

    def test(self):

        self.check(data_size='subset')
        self.check(data_size='fullset')

    @measure_time
    def check(self, data_size):

        print('=' * 50)
        print('  data_size: %s' % str(data_size))
        print('=' * 50)

        for data_type in ['train', 'eval1', 'eval2', 'eval3']:
            if data_type == 'train':
                label_paths = path.trans(data_type='train_' + data_size)
            else:
                label_paths = path.trans(data_type=data_type)
            save_vocab_file = True if data_type == 'train' else False
            is_test = True if 'eval' in data_type else False

            print('---------- %s ----------' % data_type)
            read_sdb(
                label_paths=label_paths,
                data_size=data_size,
                vocab_file_save_path=mkdir_join('../config', 'vocab_files'),
                save_vocab_file=save_vocab_file,
                is_test=is_test,
                data_type=data_type)


if __name__ == '__main__':
    unittest.main()
