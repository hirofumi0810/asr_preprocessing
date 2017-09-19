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
from librispeech.labels.word import read_text


class TestCTCLabelWord(unittest.TestCase):

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
        self.check_reading(model='ctc', train_data_size='train_clean100')
        self.check_reading(model='ctc', train_data_size='train_clean360')
        self.check_reading(model='ctc', train_data_size='train_other500')
        self.check_reading(model='ctc', train_data_size='train_all')

        # Attention
        self.check_reading(model='attention', train_data_size='train_clean100')
        self.check_reading(model='attention', train_data_size='train_clean360')
        self.check_reading(model='attention', train_data_size='train_other500')
        self.check_reading(model='attention', train_data_size='train_all')

    def check_reading(self, model, train_data_size):

        print('==================================================')
        print('  model: %s' % model)
        print('  train_data_size: %s' % str(train_data_size))
        print('==================================================')

        print('---------- train ----------')
        label_train_paths = self.prep.text(data_type=train_data_size)
        read_text(label_paths=label_train_paths,
                  data_type=train_data_size,
                  train_data_size=train_data_size,
                  run_root_path=self.prep.run_root_path,
                  model=model,
                  save_map_file=True,
                  frequency_threshold=10,
                  stdout_transcript=False)

        for data_type in ['dev_clean', 'dev_other', 'test_clean', 'test_other']:

            print('---------- %s ----------' % data_type)
            read_text(label_paths=self.label_paths[data_type],
                      data_type=data_type,
                      train_data_size=train_data_size,
                      run_root_path=self.prep.run_root_path,
                      model=model,
                      stdout_transcript=False)


if __name__ == '__main__':
    unittest.main()
