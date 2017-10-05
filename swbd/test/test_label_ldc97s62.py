#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys
import unittest
from glob import glob

sys.path.append('../../')
from swbd.labels.ldc97s62.character import read_char
from utils.measure_time_func import measure_time

swbd_trans_path = '/n/sd8/inaguma/corpus/swbd/dataset/swb_ms98_transcriptions'

# Search paths to transcript
label_paths = []
for trans_path in glob(join(swbd_trans_path, '*/*/*.text')):
    if trans_path.split('.')[0][-5:] == 'trans':
        label_paths.append(trans_path)


class TestCTCLabelSwitchboardChar(unittest.TestCase):

    def test(self):

        self.check_reading()

    @measure_time
    def check_reading(self):

        read_char(label_paths=sorted(label_paths),
                  run_root_path='../',
                  frequency_threshold=5)


if __name__ == '__main__':
    unittest.main()
