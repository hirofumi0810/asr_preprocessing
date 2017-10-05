#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from timit.path import Path
from timit.labels.phone import read_phone
from utils.measure_time_func import measure_time

path = Path(data_path='/n/sd8/inaguma/corpus/timit/original',
            run_root_path='../')

label_paths = {
    'train': path.phone(data_type='train'),
    'dev': path.phone(data_type='dev'),
    'test': path.phone(data_type='test')
}


class TestEnd2EndLabelPhone(unittest.TestCase):

    def test(self):

        self.check_reading()

    @measure_time
    def check_reading(self):

        for data_type in ['train', 'dev', 'test']:
            save_map_file = True if data_type == 'train' else False

            print('---------- %s ----------' % data_type)
            read_phone(label_paths=label_paths[data_type],
                       run_root_path=path.run_root_path,
                       save_map_file=save_map_file)


if __name__ == '__main__':
    unittest.main()
