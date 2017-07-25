#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append('../../')
from csj.prepare_path import Prepare
from csj.labels.attention.character import read_sdb


class TestAttentionLabel(unittest.TestCase):

    def test(self):
        self.check_reading()

    def check_reading(self):

        print('===== Attention label test =====')

        prep = Prepare(data_path='/n/sd8/inaguma/corpus/csj/data/',
                       run_root_path=os.path.abspath('../'))
        label_train_fullset_paths = prep.trans(data_type='train_fullset')
        label_train_subset_paths = prep.trans(data_type='train_subset')
        label_eval1_paths = prep.trans(data_type='eval1')
        label_eval2_paths = prep.trans(data_type='eval2')
        label_eval3_paths = prep.trans(data_type='eval3')

        print('---------- train_fullset ----------')
        read_sdb(label_paths=label_train_fullset_paths,
                 run_root_path=prep.run_root_path, save_map_file=True)
        print('---------- train_subset ----------')
        read_sdb(label_paths=label_train_subset_paths,
                 run_root_path=prep.run_root_path)
        print('---------- eval1 ----------')
        read_sdb(label_paths=label_eval1_paths,
                 run_root_path=prep.run_root_path)
        print('---------- eval2 ----------')
        read_sdb(label_paths=label_eval2_paths,
                 run_root_path=prep.run_root_path)
        print('---------- eval3 ----------')
        read_sdb(label_paths=label_eval3_paths,
                 run_root_path=prep.run_root_path)


if __name__ == '__main__':
    unittest.main()
