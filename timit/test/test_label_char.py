#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test for character-level transcript (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import abspath
import sys
import unittest

sys.path.append('../../')
from timit.path import Path
from timit.transcript_character import read_char
from utils.measure_time_func import measure_time

path = Path(data_path='/n/sd8/inaguma/corpus/timit/data',
            config_path='../config')

label_paths = {
    'train': path.trans(data_type='train'),
    'dev': path.trans(data_type='dev'),
    'test': path.trans(data_type='test')
}


class TestLabelChar(unittest.TestCase):

    def test(self):

        self.check()

    @measure_time
    def check(self):

        for data_type in ['train', 'dev', 'test']:
            save_map_file = True if data_type == 'train' else False
            is_test = True if data_type == 'test' else False

            print('---------- %s ----------' % data_type)
            read_char(label_paths=label_paths[data_type],
                      map_file_save_path=abspath('../config/mapping_files'),
                      is_test=is_test,
                      save_map_file=save_map_file)


if __name__ == '__main__':
    unittest.main()
