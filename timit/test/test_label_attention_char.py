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
from timit.labels.attention.character import read_text


class TestAttentionLabelChar(unittest.TestCase):

    def test(self):
        self.check_reading(divide_by_capital=False)
        self.check_reading(divide_by_capital=True)

    def check_reading(self, divide_by_capital):

        print('===== Attention label test (character) =====')

        prep = Prepare(data_path='/n/sd8/inaguma/corpus/timit/original/',
                       run_root_path=os.path.abspath('../'))
        label_train_paths = prep.text(data_type='train')
        label_dev_paths = prep.text(data_type='dev')
        label_test_paths = prep.text(data_type='test')

        read_text(label_paths=label_train_paths,
                  run_root_path=prep.run_root_path,
                  save_map_file=True,
                  divide_by_capital=divide_by_capital)
        read_text(label_paths=label_dev_paths,
                  run_root_path=prep.run_root_path,
                  divide_by_capital=divide_by_capital)
        read_text(label_paths=label_test_paths,
                  run_root_path=prep.run_root_path,
                  divide_by_capital=divide_by_capital)


if __name__ == '__main__':
    unittest.main()
