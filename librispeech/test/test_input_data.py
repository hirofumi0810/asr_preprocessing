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
from librispeech.inputs.input_data import read_audio

prep = Prepare(
    data_path='/n/sd8/inaguma/corpus/librispeech/data',
    run_root_path=abspath('../'))

htk_save_path = '/n/sd8/inaguma/corpus/librispeech/htk'

CONFIG = {
    'feature_type': 'logmelfbank',
    'channels': 40,
    'sampling_rate': 8000,
    'window': 0.025,
    'slide': 0.01,
    'energy': True,
    'delta': True,
    'deltadelta': True
}


class TestInputSpeakerNorm(unittest.TestCase):

    def test(self):

        self.check_reading(normalize='global', tool='htk')
        self.check_reading(normalize='speaker', tool='htk')
        self.check_reading(normalize='utterance', tool='htk')

        self.check_reading(normalize='global', tool='python_speech_features')
        self.check_reading(normalize='speaker', tool='python_speech_features')
        self.check_reading(normalize='utterance', tool='python_speech_features')

        self.check_reading(normalize='global', tool='librosa')
        self.check_reading(normalize='speaker', tool='librosa')
        self.check_reading(normalize='utterance', tool='librosa')

    def check_reading(self, normalize, tool):

        print('==================================================')
        print('  normalize: %s' % normalize)
        print('  tool: %s' % tool)
        print('==================================================')

        if tool == 'htk':
            audio_paths = {
                'train': [path for path in sorted(glob(
                    join(htk_save_path, 'train_clean100/*/*/*.htk')))],
                'dev_clean': [path for path in sorted(glob(
                    join(htk_save_path, 'dev_clean/*/*/*.htk')))],
                'dev_other': [path for path in sorted(glob(
                    join(htk_save_path, 'dev_other/*/*/*.htk')))],
                'test_clean': [path for path in sorted(glob(
                    join(htk_save_path, 'test_clean/*/*/*.htk')))],
                'test_other': [path for path in sorted(glob(
                    join(htk_save_path, 'test_other/*/*/*.htk')))],
            }
        else:
            audio_paths = {
                'train': prep.wav(data_type='train'),
                'dev_clean': prep.wav(data_type='dev_clean'),
                'dev_other': prep.wav(data_type='_otherdev'),
                'test_clean': prep.wav(data_type='test_clean'),
                'test_other': prep.wav(data_type='test_other')
            }

        print('---------- train ----------')
        train_global_mean_male, train_global_mean_female, train_global_std_male, train_global_std_female = read_audio(
            audio_paths=audio_paths['train'],
            tool=tool,
            config=CONFIG,
            normalize=normalize,
            is_training=True,
            speaker_gender_dict=prep.speaker_gender_dict)

        for data_type in ['dev_clean', 'dev_other', 'test_clean', 'test_other']:

            print('---------- %s ----------' % data_type)
            read_audio(audio_paths=audio_paths[data_type],
                       tool=tool,
                       config=CONFIG,
                       normalize=normalize,
                       is_training=False,
                       speaker_gender_dict=prep.speaker_gender_dict,
                       train_global_mean_male=train_global_mean_male,
                       train_global_mean_female=train_global_mean_female,
                       train_global_std_male=train_global_std_male,
                       train_global_std_female=train_global_std_female)


if __name__ == '__main__':
    unittest.main()
