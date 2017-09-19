#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append('../../')
from librispeech.prepare_path import Prepare
from librispeech.labels.character import read_text


class TestCTCLabelChar(unittest.TestCase):

    def test(self):

        self.prep = Prepare(
            data_path='/n/sd8/inaguma/corpus/librispeech/data',
            run_root_path=os.path.abspath('../'))

        self.label_paths = {
            'train_clean100': self.prep.text(data_type='train_clean100'),
            'train_clean360': self.prep.text(data_type='train_clean360'),
            'train_other500': self.prep.text(data_type='train_other500'),
            'train_all': self.prep.text(data_type='train_all'),
            'dev_clean': self.prep.text(data_type='dev_clean'),
            'dev_other': self.prep.text(data_type='dev_other'),
            'test_clean': self.prep.text(data_type='test_clean'),
            'test_other': self.prep.text(data_type='test_other')
        }

        # CTC
        self.check_reading(model='ctc', divide_by_capital=False)
        self.check_reading(model='ctc', divide_by_capital=True)

        # Attention
        self.check_reading(model='attention', divide_by_capital=False)
        self.check_reading(model='attention', divide_by_capital=True)

    def check_reading(self, model, divide_by_capital):

        print('==================================================')
        print('  model: %s' % model)
        print('  divide_by_capital: %s' % str(divide_by_capital))
        print('==================================================')

        for data_type in ['train_clean100', 'train_clean360', 'train_other500',
                          'dev_clean', 'dev_other', 'test_clean', 'test_other']:

            save_map_file = True if data_type == 'train_all' else False

            print('---------- %s ----------' % data_type)
            read_text(label_paths=self.label_paths[data_type],
                      run_root_path=self.prep.run_root_path,
                      model=model,
                      save_map_file=save_map_file,
                      divide_by_capital=divide_by_capital,
                      stdout_transcript=False)


if __name__ == '__main__':
    unittest.main()
