
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import unittest
from glob import glob

sys.path.append('../../')
from csj.prepare_path import Prepare
from csj.inputs.input_data import read_htk
from csj.labels.ctc.character import read_sdb

train_mean_male = None
train_mean_female = None
train_std_male = None
train_std_female = None


class TestInputSpeakerNorm(unittest.TestCase):

    def test(self):

        print('===== speaker norm input test =====')
        prep = Prepare(csj_path='/n/sd8/inaguma/corpus/csj/data/',
                       run_root_path=abspath('../'))
        input_feature_path = '/n/sd8/inaguma/corpus/csj/fbank/'

        for data_type in ['train', 'train_large', 'eval1', 'eval2', 'eval3']:
            print('=> ' + data_type)
            label_paths = prep.trans(data_type)
            speaker_dict = read_sdb(label_paths=label_paths,
                                    label_type='character',
                                    run_root_path=prep.run_root_path)
            htk_paths = [join(input_feature_path, htk_path)
                         for htk_path in sorted(glob(join(input_feature_path,
                                                          data_type + '/*.htk')))]

            global train_mean_male
            global train_mean_female
            global train_std_male
            global train_std_female
            if data_type in ['train', 'train_large']:
                return_tuple = read_htk(htk_paths=htk_paths,
                                        speaker_dict=speaker_dict,
                                        normalize='speaker',
                                        is_training=True)
                train_mean_male, train_mean_female, train_std_male, train_std_female = return_tuple

            else:
                read_htk(htk_paths=htk_paths,
                         speaker_dict=speaker_dict,
                         normalize='speaker',
                         is_training=False,
                         train_mean_male=train_mean_male,
                         train_mean_female=train_mean_female,
                         train_std_male=train_std_female,
                         train_std_female=train_std_female)


if __name__ == '__main__':
    unittest.main()
