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
from csj.inputs.input_data import read_audio
from csj.labels.character import read_sdb
from utils.measure_time_func import measure_time

prep = Prepare(data_path='/n/sd8/inaguma/corpus/csj/data',
               run_root_path=abspath('../'))

htk_save_path = '/n/sd8/inaguma/corpus/csj/htk'
htk_paths = {
    'train': [path for path in sorted(
        glob(join(htk_save_path, 'train_subset/*.htk')))],
    'dev': [path for path in sorted(glob(join(htk_save_path, 'dev/*.htk')))],
    'eval1': [path for path in sorted(glob(join(htk_save_path, 'eval1/*.htk')))],
    'eval2': [path for path in sorted(glob(join(htk_save_path, 'eval2/*.htk')))],
    'eval3': [path for path in sorted(glob(join(htk_save_path, 'eval3/*.htk')))]
}

wav_paths = {
    'train': prep.wav(data_type='train_subset'),
    'dev': prep.wav(data_type='dev'),
    'eval1': prep.wav(data_type='eval1'),
    'eval2': prep.wav(data_type='eval2'),
    'eval3': prep.wav(data_type='eval3'),
}

label_paths = {
    'train': prep.trans(data_type='train_subset'),
    'dev': prep.trans(data_type='dev'),
    'eval1': prep.trans(data_type='eval1'),
    'eval2': prep.trans(data_type='eval2'),
    'eval3': prep.trans(data_type='eval3'),
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

        self.check_feature_extraction(normalize='global', tool='htk')
        self.check_feature_extraction(normalize='speaker', tool='htk')
        self.check_feature_extraction(normalize='utterance', tool='htk')

        self.check_feature_extraction(
            normalize='global', tool='python_speech_features')
        self.check_feature_extraction(
            normalize='speaker', tool='python_speech_features')
        self.check_feature_extraction(
            normalize='utterance', tool='python_speech_features')

        self.check_feature_extraction(normalize='global', tool='librosa')
        self.check_feature_extraction(normalize='speaker', tool='librosa')
        self.check_feature_extraction(normalize='utterance', tool='librosa')

    @measure_time
    def check_feature_extraction(self, normalize, tool):

        print('==================================================')
        print('  normalize: %s' % normalize)
        print('  tool: %s' % tool)
        print('==================================================')

        audio_paths = htk_paths if tool == 'htk' else wav_paths

        print('---------- train ----------')
        speaker_dict = read_sdb(label_paths=label_paths['train'],
                                run_root_path=prep.run_root_path,
                                model='ctc')
        train_global_mean_male, train_global_mean_female, train_global_std_male, train_global_std_female = read_audio(
            audio_paths=audio_paths['train'],
            tool=tool,
            config=CONFIG,
            speaker_dict=speaker_dict,
            normalize=normalize,
            is_training=True)

        for data_type in ['dev', 'eval1', 'eval2', 'eval3']:
            print('---------- %s ----------' % data_type)
            speaker_dict = read_sdb(label_paths=label_paths[data_type],
                                    run_root_path=prep.run_root_path,
                                    model='ctc')
            read_audio(audio_paths=audio_paths[data_type],
                       tool=tool,
                       config=CONFIG,
                       speaker_dict=speaker_dict,
                       normalize=normalize,
                       is_training=False,
                       train_global_mean_male=train_global_mean_male,
                       train_global_mean_female=train_global_mean_female,
                       train_global_std_male=train_global_std_male,
                       train_global_std_female=train_global_std_female)


if __name__ == '__main__':
    unittest.main()
