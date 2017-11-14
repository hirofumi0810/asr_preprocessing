#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test for input data (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from librispeech.path import Path
from librispeech.input_data import read_audio
from utils.measure_time_func import measure_time

path = Path(
    data_path='/n/sd8/inaguma/corpus/librispeech/data',
    htk_save_path='/n/sd8/inaguma/corpus/librispeech/htk')

htk_paths = {
    'train100h': path.htk(data_type='train100h'),
    'dev_clean': path.htk(data_type='dev_clean'),
    'dev_other': path.htk(data_type='dev_other'),
    'test_clean': path.htk(data_type='test_clean'),
    'test_other': path.htk(data_type='test_other')
}

wav_paths = {
    'train100h': path.wav(data_type='train100h'),
    'dev_clean': path.wav(data_type='dev_clean'),
    'dev_other': path.wav(data_type='dev_other'),
    'test_clean': path.wav(data_type='test_clean'),
    'test_other': path.wav(data_type='test_other')
}

CONFIG = {
    'feature_type': 'logmelfbank',
    'channels': 40,
    'sampling_rate': 16000,
    'window': 0.025,
    'slide': 0.01,
    'energy': False,
    'delta': True,
    'deltadelta': True
}


class TestInput(unittest.TestCase):

    def test(self):

        self.check(normalize='global', tool='htk')
        self.check(normalize='speaker', tool='htk')
        self.check(normalize='utterance', tool='htk')

        self.check(normalize='global', tool='python_speech_features')
        self.check(normalize='speaker', tool='python_speech_features')
        self.check(normalize='utterance', tool='python_speech_features')

        self.check(normalize='global', tool='librosa')
        self.check(normalize='speaker', tool='librosa')
        self.check(normalize='utterance', tool='librosa')

    @measure_time
    def check(self, normalize, tool):

        print('==================================================')
        print('  normalize: %s' % normalize)
        print('  tool: %s' % tool)
        print('==================================================')

        audio_paths = htk_paths if tool == 'htk' else wav_paths

        print('---------- train100h ----------')
        global_mean_male, global_mean_female, global_std_male, global_std_female = read_audio(
            audio_paths=audio_paths['train100h'],
            tool=tool,
            config=CONFIG,
            normalize=normalize,
            is_training=True,
            speaker_gender_dict=path.speaker_gender_dict)

        for data_type in ['dev_clean', 'dev_other', 'test_clean', 'test_other']:

            print('---------- %s ----------' % data_type)
            read_audio(
                audio_paths=audio_paths[data_type],
                tool=tool,
                config=CONFIG,
                normalize=normalize,
                is_training=False,
                speaker_gender_dict=path.speaker_gender_dict,
                global_mean_male=global_mean_male,
                global_mean_female=global_mean_female,
                global_std_male=global_std_male,
                global_std_female=global_std_female)


if __name__ == '__main__':
    unittest.main()
