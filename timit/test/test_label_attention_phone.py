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
from labels.attention.phone import read_phone


class TestAttentionLabelPhone(unittest.TestCase):
    def test(self):

        prep = Prepare(timit_path='/n/sd8/inaguma/corpus/timit/original/',
                       run_root_path=os.path.abspath('../'))
        label_train_paths = prep.phone(data_type='train')
        label_dev_paths = prep.phone(data_type='dev')
        label_test_paths = prep.phone(data_type='test')

        print('===== phone61 =====')
        read_phone(label_paths=label_train_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone61')
        read_phone(label_paths=label_dev_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone61')
        read_phone(label_paths=label_test_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone61')

        print('===== phone48 =====')
        read_phone(label_paths=label_train_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone48')
        read_phone(label_paths=label_dev_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone48')
        read_phone(label_paths=label_test_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone48')

        print('===== phone39 =====')
        read_phone(label_paths=label_train_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone39')
        read_phone(label_paths=label_dev_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone39')
        read_phone(label_paths=label_test_paths,
                   run_root_path=os.path.abspath('../'),
                   label_type='phone39')


if __name__ == '__main__':
    unittest.main()
