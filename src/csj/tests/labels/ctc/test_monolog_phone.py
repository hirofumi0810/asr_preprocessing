#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../../../')
sys.path.append('../../../../')
from prepare_path import Prepare
from labels.ctc.monolog.character import read_sdb


class TestCTCLabelMonologPhone(unittest.TestCase):

    def test(self):
        prep = Prepare()
        # label_train_paths = prep.trans(data_type='train')
        label_train_all_paths = prep.trans(data_type='train_all')
        label_eval1_paths = prep.trans(data_type='eval1')
        label_eval2_paths = prep.trans(data_type='eval2')
        label_eval3_paths = prep.trans(data_type='eval3')

        print('===== monolog (phone) =====')
        # read_sdb(label_paths=label_train_paths, label_type='phone')
        read_sdb(label_paths=label_train_all_paths, label_type='phone')
        read_sdb(label_paths=label_eval1_paths, label_type='phone')
        read_sdb(label_paths=label_eval2_paths, label_type='phone')
        read_sdb(label_paths=label_eval3_paths, label_type='phone')


if __name__ == '__main__':
    unittest.main()
