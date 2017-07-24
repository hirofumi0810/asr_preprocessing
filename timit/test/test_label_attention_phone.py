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
from timit.labels.attention.phone import read_phone


class TestAttentionLabelPhone(unittest.TestCase):

    def test(self):
        print('===== Attention label test (phone) =====')

        self.prep = Prepare(timit_path='/n/sd8/inaguma/corpus/timit/original/',
                            run_root_path=os.path.abspath('../'))
        self.label_train_paths = self.prep.phone(data_type='train')
        self.label_dev_paths = self.prep.phone(data_type='dev')
        self.label_test_paths = self.prep.phone(data_type='test')

        self.check_reading(label_type='phone61')
        self.check_reading(label_type='phone48')
        self.check_reading(label_type='phone39')

    def check_reading(self, label_type):

        print('===== ' + label_type + ' =====')
        read_phone(label_paths=self.label_train_paths,
                   run_root_path=self.prep.run_root_path,
                   label_type=label_type,
                   save_map_file=True)
        read_phone(label_paths=self.label_dev_paths,
                   run_root_path=self.prep.run_root_path,
                   label_type=label_type)
        read_phone(label_paths=self.label_test_paths,
                   run_root_path=self.prep.run_root_path,
                   label_type=label_type)


if __name__ == '__main__':
    unittest.main()
