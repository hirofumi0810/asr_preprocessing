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
from librispeech.labels.ctc.word import read_text


class TestCTCLabelWord(unittest.TestCase):

    def test(self):
        
        print('=================================')
        print('=     CTC label test (word)     =')
        print('=================================')

        self.prep = Prepare(
            data_path='/n/sd8/inaguma/corpus/librispeech/data/',
            run_root_path=os.path.abspath('../'))
        self.label_dev_clean_paths = self.prep.text(data_type='dev_clean')
        self.label_dev_other_paths = self.prep.text(data_type='dev_other')
        self.label_test_clean_paths = self.prep.text(data_type='test_clean')
        self.label_test_other_paths = self.prep.text(data_type='test_other')

        self.check_reading(train_data_size='train_clean100')
        self.check_reading(train_data_size='train_clean360')
        self.check_reading(train_data_size='train_other500')
        self.check_reading(train_data_size='train_all')

    def check_reading(self, train_data_size):

        print('<<<<<<<<<< train_data_size: %s >>>>>>>>>>' % train_data_size)

        print('---------- train ----------')
        label_train_paths = self.prep.text(data_type=train_data_size)
        read_text(label_paths=label_train_paths,
                  data_type=train_data_size,
                  train_data_size=train_data_size,
                  run_root_path=self.prep.run_root_path,
                  save_map_file=True,
                  frequency_threshold=10)

        print('---------- dev_clean ----------')
        read_text(label_paths=self.label_dev_clean_paths,
                  data_type='dev_clean',
                  train_data_size=train_data_size,
                  run_root_path=self.prep.run_root_path)

        print('---------- dev_other ----------')
        read_text(label_paths=self.label_dev_other_paths,
                  data_type='dev_other',
                  train_data_size=train_data_size,
                  run_root_path=self.prep.run_root_path)

        print('---------- test_clean ----------')
        read_text(label_paths=self.label_test_clean_paths,
                  data_type='test_clean',
                  train_data_size=train_data_size,
                  run_root_path=self.prep.run_root_path)

        print('---------- test_other ----------')
        read_text(label_paths=self.label_test_other_paths,
                  data_type='test_other',
                  train_data_size=train_data_size,
                  run_root_path=self.prep.run_root_path)


if __name__ == '__main__':
    unittest.main()
