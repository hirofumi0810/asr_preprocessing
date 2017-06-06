#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../../../')
sys.path.append('../../../../')
from prepare_path import Prepare
from labels.ctc.eval2000.swbd import read_txt


class TestCTCLabelEval2000(unittest.TestCase):
    def test(self):
        prep = Prepare()
        label_paths = prep.label_test('swbd')

        print('===== eval2000 =====')
        read_txt(label_paths=label_paths)


if __name__ == '__main__':
    unittest.main()
