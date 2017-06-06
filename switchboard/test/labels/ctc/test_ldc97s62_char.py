#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../../../')
sys.path.append('../../../../')
from prepare_path import Prepare
from labels.ctc.ldc97s62.character import read_trans


class TestCTCLabelLDC97S62Char(unittest.TestCase):
    def test(self):
        prep = Prepare()
        label_paths = prep.label_train(label_type='character',
                                       train_type='ldc97s62')

        print('===== ldc97s62 (character) =====')
        read_trans(label_paths=label_paths)


if __name__ == '__main__':
    unittest.main()
