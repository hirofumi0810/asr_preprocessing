#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
from glob import glob

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.ctc.monolog.character import read_sdb


class TestInputMonologGlobalNorm(unittest.TestCase):

    def test(self):
        prep = Prepare()
        data_type_list = ['train', 'train_plus',
                          'eval1', 'eval2', 'eval3']
        for data_type in data_type_list:
            label_paths = prep.trans(data_type)

            print('===== global norm (' + data_type + ') =====')
            speaker_dict = read_sdb(
                label_paths=label_paths, label_type='character')
            htk_paths = [os.path.join(prep.fbank_path, htk_path)
                         for htk_path in sorted(glob(os.path.join(prep.fbank_path,
                                                                  data_type + '/*.htk')))]
            is_training = True if data_type in [
                'train', 'train_plus'] else False
            read_htk(htk_paths=htk_paths,
                     speaker_dict=speaker_dict,
                     global_norm=True,
                     is_training=is_training)


if __name__ == '__main__':
    unittest.main()
