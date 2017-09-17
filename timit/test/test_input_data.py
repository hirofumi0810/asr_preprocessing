#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
from glob import glob
import unittest

sys.path.append('../../')
from timit.prepare_path import Prepare
from timit.inputs.input_data import read_audio


class TestInputNorm(unittest.TestCase):

    def test(self):

        self.config = {
            'feature_type': 'logmelfbank',
            'channels': 40,
            'sampling_rate': 8000,
            'window': 0.025,
            'slide': 0.01,
            'energy': True,
            'delta': True,
            'deltadelta': True
        }

        self.check_feature_extraction(tool='htk', normalize='global')
        self.check_feature_extraction(tool='htk', normalize='speaker')
        self.check_feature_extraction(tool='htk', normalize='utterance')

        # NOTE: these are very slow
        self.check_feature_extraction(tool='python_speech_features', normalize='global')
        self.check_feature_extraction(tool='python_speech_features', normalize='speaker')
        self.check_feature_extraction(tool='python_speech_features', normalize='utterance')

        self.check_feature_extraction(tool='librosa', normalize='global')
        self.check_feature_extraction(tool='librosa', normalize='speaker')
        self.check_feature_extraction(tool='librosa', normalize='utterance')

    def check_feature_extraction(self, tool, normalize):

        print('==================================================')
        print('  tool: %s' % tool)
        print('  normalize: %s' % normalize)
        print('==================================================')

        htk_save_path = '/n/sd8/inaguma/corpus/timit/htk'
        prep = Prepare('/n/sd8/inaguma/corpus/timit/original', abspath('../'))

        if tool == 'htk':
            wav_paths = {
                'train': [path for path in glob(join(htk_save_path, 'train/*/*.htk'))],
                'dev': [path for path in glob(join(htk_save_path, 'dev/*/*.htk'))],
                'test': [path for path in glob(join(htk_save_path, 'test/*/*.htk'))]
            }
            # NOTE: these are htk file paths
        else:
            wav_paths = {
                'train': prep.wav(data_type='train'),
                'dev': prep.wav(data_type='dev'),
                'test': prep.wav(data_type='test')
            }

        print('---------- train ----------')
        train_global_mean_male, train_global_std_male, train_global_mean_female, train_global_std_female = read_audio(
            audio_paths=wav_paths['train'],
            tool=tool,
            config=self.config,
            normalize=normalize,
            is_training=True)

        for data_type in ['dev', 'test']:
            print('---------- %s ----------' % data_type)
            read_audio(audio_paths=wav_paths[data_type],
                       tool=tool,
                       config=self.config,
                       normalize=normalize,
                       is_training=False,
                       train_global_mean_male=train_global_mean_male,
                       train_global_std_male=train_global_std_male,
                       train_global_mean_female=train_global_mean_female,
                       train_global_std_female=train_global_std_female)


if __name__ == '__main__':
    unittest.main()
