#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from labels.ctc.eval2000.swbd import read_transcript


class TestCTCLabelEval2000(unittest.TestCase):
    def test(self):
        prep = Prepare()
        label_test_paths = prep.label_test()
        read_transcript(label_paths=label_test_paths)


if __name__ == '__main__':
    unittest.main()
