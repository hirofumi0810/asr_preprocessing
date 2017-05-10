#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from labels.ctc.ldc97s62.character import read_transcript


class TestCTCLabelLDC97S62Character(unittest.TestCase):
    def test(self):
        prep = Prepare()
        label_train_paths = prep.label_train(label_type='character')
        read_transcript(label_paths=label_train_paths)


if __name__ == '__main__':
    unittest.main()
