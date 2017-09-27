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
from timit.labels.character import read_text
from utils.measure_time_func import measure_time

prep = Prepare(data_path='/n/sd8/inaguma/corpus/timit/original',
               run_root_path=os.path.abspath('../'))

label_paths = {
    'train': prep.text(data_type='train'),
    'dev': prep.text(data_type='dev'),
    'test': prep.text(data_type='test')
}


class TestEnd2EndLabelChar(unittest.TestCase):

    def test(self):

        # CTC
        self.check_reading(model='ctc', divide_by_capital=False)
        self.check_reading(model='ctc', divide_by_capital=True)

        # Attention
        self.check_reading(model='attention', divide_by_capital=False)
        self.check_reading(model='attention', divide_by_capital=True)

    @measure_time
    def check_reading(self, model, divide_by_capital):

        print('==================================================')
        print('  model: %s' % model)
        print('  divide_by_capital: %s' % str(divide_by_capital))
        print('==================================================')

        for data_type in ['train', 'dev', 'test']:
            save_map_file = True if data_type == 'train' else False

            print('---------- %s ----------' % data_type)
            read_text(label_paths=label_paths[data_type],
                      run_root_path=prep.run_root_path,
                      model=model,
                      save_map_file=save_map_file,
                      divide_by_capital=divide_by_capital,
                      stdout_transcript=True)


if __name__ == '__main__':
    unittest.main()
