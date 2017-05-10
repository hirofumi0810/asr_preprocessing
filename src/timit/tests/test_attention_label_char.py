#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from labels.attention.character import read_text


class TestAttentionCharacter(unittest.TestCase):
    def test(self):
        prep = Prepare()
        label_train_paths = prep.text(data_type='train')
        label_dev_paths = prep.text(data_type='dev')
        label_test_paths = prep.text(data_type='test')

        read_text(label_paths=label_train_paths)
        read_text(label_paths=label_dev_paths)
        read_text(label_paths=label_test_paths)


if __name__ == '__main__':
    unittest.main()
