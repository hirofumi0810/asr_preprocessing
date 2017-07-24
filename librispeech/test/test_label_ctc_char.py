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
from librispeech.labels.ctc.character import read_text


class TestCTCLabelChar(unittest.TestCase):

    def test(self):
        self.check_reading(divide_by_capital=False)
        self.check_reading(divide_by_capital=True)

    def check_reading(self, divide_by_capital):

        print('===== CTC label test (character) =====')

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
                  run_root_path=prep.run_root_path,
                  divide_by_capital=divide_by_capital)
        read_text(label_paths=label_train_clean360_paths,
                  run_root_path=prep.run_root_path,
                  divide_by_capital=divide_by_capital)
        read_text(label_paths=label_train_other500_paths,
                  run_root_path=prep.run_root_path,
                  save_map_file=True,
                  divide_by_capital=divide_by_capital)

        # dev
        read_text(label_paths=label_dev_clean_paths,
                  run_root_path=prep.run_root_path,
                  divide_by_capital=divide_by_capital)
        read_text(label_paths=label_dev_other_paths,
                  run_root_path=prep.run_root_path,
                  divide_by_capital=divide_by_capital)

        # test
        read_text(label_paths=label_test_clean_paths,
                  run_root_path=prep.run_root_path,
                  divide_by_capital=divide_by_capital)
        read_text(label_paths=label_test_other_paths,
                  run_root_path=prep.run_root_path,
                  divide_by_capital=divide_by_capital)


if __name__ == '__main__':
    unittest.main()
