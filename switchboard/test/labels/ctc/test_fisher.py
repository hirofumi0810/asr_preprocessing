#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../../../')
sys.path.append('../../../../')
from prepare_path import Prepare
from labels.ctc.fisher.character import read_trans


class TestCTCLabelFisherChar(unittest.TestCase):
    def test(self):
        prep = Prepare()
        label_paths = prep.label_train(label_type='character',
                                       train_type='fisher')

        print('===== fisher (character) =====')
        read_trans(label_paths=label_paths, speaker='A')
        read_trans(label_paths=label_paths, speaker='B')


if __name__ == '__main__':
    unittest.main()
