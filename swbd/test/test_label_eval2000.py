#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename, join
import sys
import unittest
from glob import glob

sys.path.append('../../')
from swbd.labels.eval2000.swbd_text import read_text
from swbd.labels.eval2000.swbd_ch_stm import read_stm
from utils.measure_time_func import measure_time

eval2000_trans_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43'
eval2000_stm_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/hub5e00.english.000405.stm'
eval2000_pem_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002S09/english/hub5e_00.pem'

label_paths = []
for file_path in glob(join(eval2000_trans_path, 'reference/english/*')):
    if basename(file_path)[:2] == 'sw':
        label_paths.append(file_path)


class TestLabelEval2000Swbd(unittest.TestCase):

    def test(self):

        self.check_reading()

    @measure_time
    def check_reading(self):

        read_text(label_paths=sorted(label_paths),
                  pem_path=eval2000_pem_path,
                  run_root_path='../')

        read_stm(stm_path=eval2000_stm_path,
                 pem_path=eval2000_pem_path,
                 run_root_path='../')


if __name__ == '__main__':
    unittest.main()
