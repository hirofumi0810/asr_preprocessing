#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from labels.ctc.character import read_sdb


class TestCTCLabel(unittest.TestCase):

    def test(self):
        self.check_reading(label_type='kanji')
        self.check_reading(label_type='character')
        self.check_reading(label_type='phone')

    def check_reading(self, label_type):

        print('===== CTC label test (' + label_type + ') =====')

        prep = Prepare(csj_path='/n/sd8/inaguma/corpus/csj/data/',
                       run_root_path=os.path.abspath('../'))
        label_train_paths = prep.trans(data_type='train')
        label_train_large_paths = prep.trans(data_type='train_large')
        label_eval1_paths = prep.trans(data_type='eval1')
        label_eval2_paths = prep.trans(data_type='eval2')
        label_eval3_paths = prep.trans(data_type='eval3')

        print('===== ' + label_type + ' =====')
        read_sdb(label_paths=label_train_large_paths, label_type=label_type,
                 run_root_path=prep.run_root_path, save_map_file=True)
        read_sdb(label_paths=label_train_paths, label_type=label_type,
                 run_root_path=prep.run_root_path)
        read_sdb(label_paths=label_eval1_paths, label_type=label_type,
                 run_root_path=prep.run_root_path)
        read_sdb(label_paths=label_eval2_paths, label_type=label_type,
                 run_root_path=prep.run_root_path)
        read_sdb(label_paths=label_eval3_paths, label_type=label_type,
                 run_root_path=prep.run_root_path)


if __name__ == '__main__':
    unittest.main()
