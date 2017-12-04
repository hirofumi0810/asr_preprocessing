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
from swbd.labels.ldc97s62.character import read_trans
from utils.measure_time_func import measure_time
from utils.util import mkdir_join

swbd_trans_path = '/n/sd8/inaguma/corpus/swbd/swb_ms98_transcriptions'

# Search paths to transcript
label_paths = []
for trans_path in glob(join(swbd_trans_path, '*/*/*.text')):
    if trans_path.split('.')[0][-5:] == 'trans':
        label_paths.append(trans_path)
label_paths = sorted(label_paths)

wb_paths = []
for wb_path in glob(join(swbd_trans_path, '*/*/*.text')):
    if wb_path.split('.')[0][-4:] == 'word':
        wb_paths.append(wb_path)
wb_paths = sorted(wb_paths)


class TestLabelLDC97S62(unittest.TestCase):

    def test(self):

        self.check()

    @measure_time
    def check(self):

        read_trans(
            label_paths=label_paths,
            word_boundary_paths=wb_paths,
            run_root_path='../',
            vocab_file_save_path=mkdir_join('../config/vocab_files'),
            save_vocab_file=True)


if __name__ == '__main__':
    unittest.main()
