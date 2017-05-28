
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

train_mean_male = None
train_mean_female = None
train_std_male = None
train_std_female = None


class TestInputMonologSpeakerNorm(unittest.TestCase):

    def test(self):
        prep = Prepare()
        data_type_list = ['train', 'train_all',
                          'eval1', 'eval2', 'eval3']
        for data_type in data_type_list:
            label_paths = prep.trans(data_type)

            print('===== speaker norm (' + data_type + ') =====')
            speaker_dict = read_sdb(
                label_paths=label_paths, label_type='character')
            htk_paths = [os.path.join(prep.fbank_path, htk_path)
                         for htk_path in sorted(glob(os.path.join(prep.fbank_path,
                                                                  data_type + '/*.htk')))]

            global train_mean_male
            global train_mean_female
            global train_mean_male
            global train_mean_female
            if data_type in ['train', 'train_all']
                return_tuple = read_htk(htk_paths=htk_paths,
                                        speaker_dict=speaker_dict,
                                        normalize='speaker',
                                        is_training=True,
                                        train_mean_male=train_mean_male,
                                        train_mean_female=train_mean_female,
                                        train_std_male=train_std_female,
                                        train_std_female=train_std_female)
            else:
                return_tuple = read_htk(htk_paths=htk_paths,
                                        speaker_dict=speaker_dict,
                                        normalize='speaker',
                                        is_training=False,
                                        train_mean_male=train_mean_male,
                                        train_mean_female=train_mean_female,
                                        train_std_male=train_std_female,
                                        train_std_female=train_std_female)
            train_mean_male, train_mean_female, train_std_male, train_std_female = return_tuple


if __name__ == '__main__':
    unittest.main()
