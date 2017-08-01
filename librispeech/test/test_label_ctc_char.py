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

        print('======================================')
        print('=     CTC label test (character)     =')
        print('======================================')

        self.prep = Prepare(
            data_path='/n/sd8/inaguma/corpus/librispeech/data/',
            run_root_path=os.path.abspath('../'))
        self.label_train_clean100_paths = self.prep.text(
            data_type='train_clean100')
        self.label_train_clean360_paths = self.prep.text(
            data_type='train_clean360')
        self.label_train_other500_paths = self.prep.text(
            data_type='train_other500')
        self.label_train_all_paths = self.prep.text(
            data_type='train_all')

        self.label_dev_clean_paths = self.prep.text(data_type='dev_clean')
        self.label_dev_other_paths = self.prep.text(data_type='dev_other')
        self.label_test_clean_paths = self.prep.text(data_type='test_clean')
        self.label_test_other_paths = self.prep.text(data_type='test_other')

        self.check_reading(divide_by_capital=False)
        self.check_reading(divide_by_capital=True)

    def check_reading(self, divide_by_capital):

        print('<<<<<<<<<< divide_by_capital: %s >>>>>>>>>>' %
              str(divide_by_capital))

        print('---------- train_clean100 ----------')
        read_text(label_paths=self.label_train_clean100_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)

        print('---------- train_clean360 ----------')
        read_text(label_paths=self.label_train_clean360_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)

        print('---------- train_other500 ----------')
        read_text(label_paths=self.label_train_other500_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)

        print('---------- train_all ----------')
        read_text(label_paths=self.label_train_all_paths,
                  run_root_path=self.prep.run_root_path,
                  save_map_file=True,
                  divide_by_capital=divide_by_capital)

        print('---------- dev_clean ----------')
        read_text(label_paths=self.label_dev_clean_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)

        print('---------- dev_other ----------')
        read_text(label_paths=self.label_dev_other_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)

        print('---------- test_clean ----------')
        read_text(label_paths=self.label_test_clean_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)

        print('---------- test_other ----------')
        read_text(label_paths=self.label_test_other_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)


if __name__ == '__main__':
    unittest.main()
