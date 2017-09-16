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
from timit.labels.ctc.character import read_text


class TestCTCLabelChar(unittest.TestCase):

    def test(self):

        print('======================================')
        print('=     CTC label test (character)     =')
        print('======================================')

        self.prep = Prepare(data_path='/n/sd8/inaguma/corpus/timit/original/',
                            run_root_path=os.path.abspath('../'))
        self.label_train_paths = self.prep.text(data_type='train')
        self.label_dev_paths = self.prep.text(data_type='dev')
        self.label_test_paths = self.prep.text(data_type='test')

        self.check_reading(divide_by_capital=False)
        self.check_reading(divide_by_capital=True)

    def check_reading(self, divide_by_capital):

        print('<<<<<<<<<< divide_by_capital: %s >>>>>>>>>>' %
              str(divide_by_capital))

        print('---------- train ----------')
        read_text(label_paths=self.label_train_paths,
                  run_root_path=self.prep.run_root_path,
                  save_map_file=True,
                  divide_by_capital=divide_by_capital)

        print('---------- dev ----------')
        read_text(label_paths=self.label_dev_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)

        print('---------- test ----------')
        read_text(label_paths=self.label_test_paths,
                  run_root_path=self.prep.run_root_path,
                  divide_by_capital=divide_by_capital)


if __name__ == '__main__':
    unittest.main()
