#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append('../../')
from csj.prepare_path import Prepare
from csj.labels.character import read_sdb


class TestEnd2EndLabel(unittest.TestCase):

    def test(self):

        self.prep = Prepare(data_path='/n/sd8/inaguma/corpus/csj/data',
                            run_root_path=os.path.abspath('../'))

        self.label_paths = {
            'train_fullset': self.prep.trans(data_type='train_fullset'),
            'train_subset': self.prep.trans(data_type='train_subset'),
            'dev': self.prep.trans(data_type='dev'),
            'eval1': self.prep.trans(data_type='eval1'),
            'eval2': self.prep.trans(data_type='eval2'),
            'eval3': self.prep.trans(data_type='eval3')
        }

        # CTC
        self.check_reading(model='ctc', divide_by_space=True)
        self.check_reading(model='ctc', divide_by_space=False)

        # Attention
        self.check_reading(model='attention', divide_by_space=True)
        self.check_reading(model='attention', divide_by_space=False)

    def check_reading(self, model, divide_by_space):

        print('==================================================')
        print('  model: %s' % model)
        print('  divide_by_space: %s' % str(divide_by_space))
        print('==================================================')

        print('---------- train_fullset ----------')
        read_sdb(label_paths=self.label_paths['train_fullset'],
                 run_root_path=self.prep.run_root_path,
                 model=model,
                 save_map_file=True,
                 divide_by_space=divide_by_space,
                 stdout_transcript=False)

        for data_type in ['train_subset', 'dev', 'eval1', 'eval2', 'eval3']:
            print('---------- %s ----------' % data_type)
            read_sdb(label_paths=self.label_paths[data_type],
                     run_root_path=self.prep.run_root_path,
                     model=model,
                     divide_by_space=divide_by_space)


if __name__ == '__main__':
    unittest.main()
