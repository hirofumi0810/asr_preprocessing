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
from utils.util import mkdir_join

path = Path(data_path='/n/sd8/inaguma/corpus/librispeech/data')


class TestLabel(unittest.TestCase):

    def test(self):

        self.check(data_size='100h')
        self.check(data_size='460h')
        self.check(data_size='960h')

    @measure_time
    def check(self, data_size):

        print('=' * 30)
        print('  data_size: %s' % str(data_size))
        print('=' * 30)

        print('---------- train ----------')
        read_trans(
            label_paths=path.trans(data_type='train' + data_size),
            data_size=data_size,
            vocab_file_save_path=mkdir_join('../config/vocab_files'),
            is_training=True,
            save_vocab_file=True)

        for data_type in ['dev_clean', 'dev_other']:
            print('---------- %s ----------' % data_type)
            read_trans(
                label_paths=path.trans(data_type=data_type),
                data_size=data_size,
                vocab_file_save_path=mkdir_join('../config/vocab_files'))

        for data_type in ['test_clean', 'test_other']:
            print('---------- %s ----------' % data_type)
            read_trans(
                label_paths=path.trans(data_type=data_type),
                data_size=data_size,
                vocab_file_save_path=mkdir_join('../config/vocab_files'),
                is_test=True)


if __name__ == '__main__':
    unittest.main()
