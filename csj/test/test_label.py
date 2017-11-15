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

        print('=' * 30)
        print('  data_size: %s' % str(data_size))
        print('=' * 30)

        print('---------- train ----------')
        read_sdb(
            label_paths=path.trans(data_type='train_' + data_size),
            data_size=data_size,
            vocab_file_save_path=mkdir_join('../config', 'vocab_files'),
            is_training=True,
            save_vocab_file=True)

        for data_type in ['dev', 'eval1', 'eval2', 'eval3']:
            print('---------- %s ----------' % data_type)
            read_sdb(
                label_paths=path.trans(data_type=data_type),
                data_size='subset',
                vocab_file_save_path=mkdir_join('../config', 'vocab_files'),
                is_test=True)


if __name__ == '__main__':
    unittest.main()
