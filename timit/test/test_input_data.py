#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test for input data (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from timit.path import Path
from timit.input_data import read_audio
from utils.measure_time_func import measure_time

path = Path(data_path='/n/sd8/inaguma/corpus/timit/data',
            config_path='../config',
            htk_save_path='/n/sd8/inaguma/corpus/timit/htk')

htk_paths = {
    'train': path.htk(data_type='train'),
    'dev': path.htk(data_type='dev'),
    'test': path.htk(data_type='test')
}

wav_paths = {
    'train': path.wav(data_type='train'),
    'dev': path.wav(data_type='dev'),
    'test': path.wav(data_type='test')
}

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


class TestInput(unittest.TestCase):

    def test(self):

        # self.check(tool='htk', normalize='global')
        # self.check(tool='htk', normalize='speaker')
        # self.check(tool='htk', normalize='utterance')

        # NOTE: these are very slow
        self.check(tool='python_speech_features', normalize='global')
        self.check(tool='python_speech_features', normalize='speaker')
        self.check(tool='python_speech_features', normalize='utterance')

        self.check(tool='librosa', normalize='global')
        self.check(tool='librosa', normalize='speaker')
        self.check(tool='librosa', normalize='utterance')

    @measure_time
    def check(self, tool, normalize):

        print('==================================================')
        print('  tool: %s' % tool)
        print('  normalize: %s' % normalize)
        print('==================================================')

        audio_paths = htk_paths if tool == 'htk' else wav_paths

        print('---------- train ----------')
        train_global_mean_male, train_global_std_male, train_global_mean_female, train_global_std_female = read_audio(
            audio_paths=audio_paths['train'],
            tool=tool,
            config=CONFIG,
            normalize=normalize,
            is_training=True)

        for data_type in ['dev', 'test']:
            print('---------- %s ----------' % data_type)
            read_audio(audio_paths=audio_paths[data_type],
                       tool=tool,
                       config=CONFIG,
                       normalize=normalize,
                       is_training=False,
                       train_global_mean_male=train_global_mean_male,
                       train_global_std_male=train_global_std_male,
                       train_global_mean_female=train_global_mean_female,
                       train_global_std_female=train_global_std_female)


if __name__ == '__main__':
    unittest.main()
