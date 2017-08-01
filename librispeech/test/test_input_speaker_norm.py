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
from librispeech.prepare_path import Prepare
from librispeech.inputs.input_data import read_htk


class TestInputSpeakerNorm(unittest.TestCase):

    def test(self):

        print('===================================')
        print('=     speaker norm input test     =')
        print('===================================')

        self.prep = Prepare(
            data_path='/n/sd8/inaguma/corpus/librispeech/data/',
            run_root_path=abspath('../'))
        self.input_feature_path = '/n/sd8/inaguma/corpus/librispeech/fbank/'

        self.check_reading(train_data_size='train_clean100')
        self.check_reading(train_data_size='train_clean360')
        self.check_reading(train_data_size='train_other500')
        self.check_reading(train_data_size='train_all')

    def check_reading(self, train_data_size):

        print('<<<<<<<<<< train_data_size: ' + train_data_size + ' >>>>>>>>>>')

        if train_data_size in ['train_other500', 'train_clean360']:
            dev_data_type = 'dev_clean'
            test_data_type = 'test_clean'
        else:
            dev_data_type = 'dev_other'
            test_data_type = 'test_other'

        if train_data_size == 'train_all':
            htk_train_paths = [
                join(self.input_feature_path, htk_path)
                for htk_path in sorted(glob(join(self.input_feature_path,
                                                 'train_clean100/*/*.htk')))]
            htk_train_paths += [
                join(self.input_feature_path, htk_path)
                for htk_path in sorted(glob(join(self.input_feature_path,
                                                 'train_clean360/*/*.htk')))]
            htk_train_paths += [
                join(self.input_feature_path, htk_path)
                for htk_path in sorted(glob(join(self.input_feature_path,
                                                 'train_other500/*/*.htk')))]
        else:
            htk_train_paths = [
                join(self.input_feature_path, htk_path)
                for htk_path in sorted(glob(join(self.input_feature_path,
                                                 train_data_size + '/*/*.htk')))]
        htk_dev_paths = [
            join(self.input_feature_path, htk_path)
            for htk_path in sorted(glob(join(self.input_feature_path,
                                             dev_data_type + '/*/*.htk')))]
        htk_test_paths = [
            join(self.input_feature_path, htk_path)
            for htk_path in sorted(glob(join(self.input_feature_path,
                                             test_data_type + '/*/*.htk')))]

        print('---------- train ----------')
        train_mean_male, train_mean_female, train_std_male, train_std_female = read_htk(
            htk_paths=htk_train_paths,
            normalize='speaker',
            is_training=True,
            speaker_gender_dict=self.prep.speaker_gender_dict)

        print('---------- dev ----------')
        read_htk(htk_paths=htk_dev_paths,
                 normalize='speaker',
                 is_training=False,
                 speaker_gender_dict=self.prep.speaker_gender_dict,
                 train_mean_male=train_mean_male,
                 train_mean_female=train_mean_female,
                 train_std_male=train_std_female,
                 train_std_female=train_std_female)

        print('---------- test ----------')
        read_htk(htk_paths=htk_test_paths,
                 normalize='speaker',
                 is_training=False,
                 speaker_gender_dict=self.prep.speaker_gender_dict,
                 train_mean_male=train_mean_male,
                 train_mean_female=train_mean_female,
                 train_std_male=train_std_female,
                 train_std_female=train_std_female)


if __name__ == '__main__':
    unittest.main()
