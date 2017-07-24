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
        self.check_reading()

    def check_reading(self):

        print('===== CTC label test (word) =====')

        prep = Prepare(data_path='/n/sd8/inaguma/corpus/librispeech/data/',
                       run_root_path=os.path.abspath('../'))
        label_train_clean100_paths = prep.text(data_type='train_clean100')
        label_train_clean360_paths = prep.text(data_type='train_clean360')
        label_train_other500_paths = prep.text(data_type='train_other500')
        label_dev_clean_paths = prep.text(data_type='dev_clean')
        label_dev_other_paths = prep.text(data_type='dev_other')
        label_test_clean_paths = prep.text(data_type='test_clean')
        label_test_other_paths = prep.text(data_type='test_other')

        # train
        read_text(label_paths=label_train_clean100_paths,
                  data_type='train_clean100',
                  run_root_path=prep.run_root_path,
                  save_map_file=True)
        read_text(label_paths=label_train_clean360_paths,
                  data_type='train_clean360',
                  run_root_path=prep.run_root_path,
                  save_map_file=True)
        read_text(label_paths=label_train_other500_paths,
                  data_type='train_other500',
                  run_root_path=prep.run_root_path,
                  save_map_file=True)

        # dev
        read_text(label_paths=label_dev_clean_paths,
                  data_type='dev_clean',
                  run_root_path=prep.run_root_path)
        read_text(label_paths=label_dev_other_paths,
                  data_type='dev_other',
                  run_root_path=prep.run_root_path)

        # test
        read_text(label_paths=label_test_clean_paths,
                  data_type='test_clean',
                  run_root_path=prep.run_root_path)
        read_text(label_paths=label_test_other_paths,
                  data_type='test_other',
                  run_root_path=prep.run_root_path)


if __name__ == '__main__':
    unittest.main()
