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
from swbd.labels.fisher.character import read_trans
from utils.measure_time_func import measure_time
from utils.util import mkdir_join

fisher_path = '/n/sd8/inaguma/corpus/swbd/data/fisher'

# Search paths to transcript
label_paths = []
for trans_path in glob(join(fisher_path, 'data/trans/*/*.txt')):
    label_paths.append(trans_path)
label_paths = sorted(label_paths)


class TestLabelFisher(unittest.TestCase):

    def test(self):

        self.check()

    @measure_time
    def check(self):

        read_trans(label_paths=label_paths, target_speaker='A')
        read_trans(label_paths=label_paths, target_speaker='B')


if __name__ == '__main__':
    unittest.main()
