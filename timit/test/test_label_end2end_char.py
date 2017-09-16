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


class TestEnd2EndLabelChar(unittest.TestCase):

    def test(self):

        print('=============================================')
        print('=     End-to-End label test (character)     =')
        print('=============================================')

        self.prep = Prepare(data_path='/n/sd8/inaguma/corpus/timit/original/',
                            run_root_path=os.path.abspath('../'))
        self.label_train_paths = self.prep.text(data_type='train')
        self.label_dev_paths = self.prep.text(data_type='dev')
        self.label_test_paths = self.prep.text(data_type='test')

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

        print('---------- train ----------')
        read_text(label_paths=self.label_train_paths,
                  run_root_path=self.prep.run_root_path,
                  model=model,
                  save_map_file=True,
                  divide_by_capital=divide_by_capital,
                  is_test=True)

        print('---------- dev ----------')
        read_text(label_paths=self.label_dev_paths,
                  run_root_path=self.prep.run_root_path,
                  model=model,
                  divide_by_capital=divide_by_capital,
                  is_test=True)

        print('---------- test ----------')
        read_text(label_paths=self.label_test_paths,
                  run_root_path=self.prep.run_root_path,
                  model=model,
                  divide_by_capital=divide_by_capital,
                  is_test=True)


if __name__ == '__main__':
    unittest.main()
