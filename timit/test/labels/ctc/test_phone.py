#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../../../')
sys.path.append('../../../../')
from prepare_path import Prepare
from labels.ctc.phone import read_phone


class TestCTCLabelPhone(unittest.TestCase):
    def test(self):
        prep = Prepare()
        label_train_paths = prep.phone(data_type='train')
        label_dev_paths = prep.phone(data_type='dev')
        label_test_paths = prep.phone(data_type='test')

        # phone61
        print('===== phone61 =====')
        read_phone(label_paths=label_train_paths, label_type='phone61')
        read_phone(label_paths=label_dev_paths, label_type='phone61')
        read_phone(label_paths=label_test_paths, label_type='phone61')

        # phone48
        print('===== phone48 =====')
        read_phone(label_paths=label_train_paths, label_type='phone48')
        read_phone(label_paths=label_dev_paths, label_type='phone48')
        read_phone(label_paths=label_test_paths, label_type='phone48')

        # phone39
        print('===== phone39 =====')
        read_phone(label_paths=label_train_paths, label_type='phone39')
        read_phone(label_paths=label_dev_paths, label_type='phone39')
        read_phone(label_paths=label_test_paths, label_type='phone39')


if __name__ == '__main__':
    unittest.main()
