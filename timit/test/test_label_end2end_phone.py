#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append('../../')
from timit.prepare_path import Prepare
from timit.labels.phone import read_phone
from utils.measure_time_func import measure_time

prep = Prepare(data_path='/n/sd8/inaguma/corpus/timit/original',
               run_root_path=os.path.abspath('../'))

label_paths = {
    'train': prep.phone(data_type='train'),
    'dev': prep.phone(data_type='dev'),
    'test': prep.phone(data_type='test')
}


class TestEnd2EndLabelPhone(unittest.TestCase):

    def test(self):

        # CTC
        self.check_reading(model='ctc', label_type='phone61')
        self.check_reading(model='ctc', label_type='phone48')
        self.check_reading(model='ctc', label_type='phone39')

        # Attention
        self.check_reading(model='attention', label_type='phone61')
        self.check_reading(model='attention', label_type='phone48')
        self.check_reading(model='attention', label_type='phone39')

    @measure_time
    def check_reading(self, model, label_type):

        print('==================================================')
        print('  model: %s' % model)
        print('  label_type: %s' % label_type)
        print('==================================================')

        for data_type in ['train', 'dev', 'test']:
            save_map_file = True if data_type == 'train' else False

            print('---------- %s ----------' % data_type)
            read_phone(label_paths=label_paths[data_type],
                       label_type=label_type,
                       run_root_path=prep.run_root_path,
                       model=model,
                       save_map_file=save_map_file,
                       stdout_transcript=True)


if __name__ == '__main__':
    unittest.main()
