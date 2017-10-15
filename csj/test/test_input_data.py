#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from csj.path import Path
from csj.input_data import read_audio
from csj.labels.transcript import read_sdb
from utils.measure_time_func import measure_time

path = Path(data_path='/n/sd8/inaguma/corpus/csj/data',
            config_path='../config',
            htk_save_path='/n/sd8/inaguma/corpus/csj/htk')

htk_paths = {
    'train': path.htk(data_type='train_subset'),
    'dev': path.htk(data_type='dev'),
    'eval1': path.htk(data_type='eval1'),
    'eval2': path.htk(data_type='eval2'),
    'eval3': path.htk(data_type='eval3'),
}

wav_paths = {
    'train': path.wav(data_type='train_subset'),
    'dev': path.wav(data_type='dev'),
    'eval1': path.wav(data_type='eval1'),
    'eval2': path.wav(data_type='eval2'),
    'eval3': path.wav(data_type='eval3'),
}

label_paths = {
    'train': path.trans(data_type='train_subset'),
    'dev': path.trans(data_type='dev'),
    'eval1': path.trans(data_type='eval1'),
    'eval2': path.trans(data_type='eval2'),
    'eval3': path.trans(data_type='eval3'),
}

CONFIG = {
    'feature_type': 'logmelfbank',
    'channels': 40,
    'sampling_rate': 16000,
    'window': 0.025,
    'slide': 0.01,
    'energy': True,
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

        print('---------- train ----------')
        speaker_dict = read_sdb(label_paths=label_paths['train'],
                                train_data_size='train_subset',
                                map_file_save_path='../config/mapping_files',
                                is_training=True,
                                save_map_file=True)
        train_global_mean_male, train_global_mean_female, train_global_std_male, train_global_std_female = read_audio(
            audio_paths=audio_paths['train'],
            speaker_dict=speaker_dict,
            tool=tool,
            config=CONFIG,
            normalize=normalize,
            is_training=True)

        for data_type in ['dev', 'eval1', 'eval2', 'eval3']:
            print('---------- %s ----------' % data_type)
            speaker_dict = read_sdb(
                label_paths=label_paths[data_type],
                train_data_size='train_subset',
                map_file_save_path='../config/mapping_files',
                is_test=True)
            read_audio(audio_paths=audio_paths[data_type],
                       speaker_dict=speaker_dict,
                       tool=tool,
                       config=CONFIG,
                       normalize=normalize,
                       is_training=False,
                       train_global_mean_male=train_global_mean_male,
                       train_global_mean_female=train_global_mean_female,
                       train_global_std_male=train_global_std_male,
                       train_global_std_female=train_global_std_female)


if __name__ == '__main__':
    unittest.main()
