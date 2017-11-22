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
from swbd.labels.eval2000.stm import read_stm
from utils.measure_time_func import measure_time

eval2000_trans_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43'
eval2000_stm_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/hub5e00.english.000405.stm'
eval2000_pem_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002S09/english/hub5e_00.pem'
eval2000_glm_path = '/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/en20000405_hub5.glm'

label_paths = []
for file_path in glob(join(eval2000_trans_path, 'reference/english/*')):
    if basename(file_path)[:2] == 'sw':
        label_paths.append(file_path)
label_paths = sorted(label_paths)


class TestLabelEval2000Swbd(unittest.TestCase):

    def test(self):

        self.check()

    @measure_time
    def check(self):

        print('---------- swbd ----------')
        # From a stm file
        read_stm(stm_path=eval2000_stm_path,
                 pem_path=eval2000_pem_path,
                 glm_path=eval2000_glm_path,
                 run_root_path='../',
                 data_size='300h')

        # From txt files
        read_text(label_paths=label_paths,
                  pem_path=eval2000_pem_path,
                  glm_path=eval2000_glm_path,
                  run_root_path='../',
                  data_size='300h')


if __name__ == '__main__':
    unittest.main()
