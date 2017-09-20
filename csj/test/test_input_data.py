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


class TestInput(unittest.TestCase):

    def test(self):

        self.prep = Prepare(data_path='/n/sd8/inaguma/corpus/csj/data',
                            run_root_path=abspath('../'))

        self.label_paths = {
            'train': self.prep.trans(data_type='train_subset'),
            'dev': self.prep.trans(data_type='dev'),
            'eval1': self.prep.trans(data_type='eval1'),
            'eval2': self.prep.trans(data_type='eval2'),
            'eval3': self.prep.trans(data_type='eval3'),
        }

        self.htk_save_path = '/n/sd8/inaguma/corpus/csj/htk'

        self.check_reading(normalize='global', tool='htk')
        self.check_reading(normalize='speaker', tool='htk')
        self.check_reading(normalize='utterance', tool='htk')

        self.check_reading(normalize='global', tool='python_speech_features')
        self.check_reading(normalize='speaker', tool='python_speech_features')
        self.check_reading(normalize='utterance', tool='python_speech_features')

        self.check_reading(normalize='global', tool='librosa')
        self.check_reading(normalize='speaker', tool='librosa')
        self.check_reading(normalize='utterance', tool='librosa')

    @profile
    def check_reading(self, normalize, tool):

        print('==================================================')
        print('  normalize: %s' % normalize)
        print('  tool: %s' % tool)
        print('==================================================')

        if tool == 'htk':
            audio_paths = {
                'train': [path for path in sorted(
                    glob(join(self.htk_save_path, 'train_subset/*.htk')))],
                'dev': [path for path in sorted(glob(join(self.htk_save_path, 'dev/*.htk')))],
                'eval1': [path for path in sorted(glob(join(self.htk_save_path, 'eval1/*.htk')))],
                'eval2': [path for path in sorted(glob(join(self.htk_save_path, 'eval2/*.htk')))],
                'eval3': [path for path in sorted(glob(join(self.htk_save_path, 'eval3/*.htk')))]
            }
            # NOTE: these are htk file paths
        else:
            audio_paths = {
                'train': self.prep.wav(data_type='train'),
                'dev': self.prep.wav(data_type='dev'),
                'eval1': self.prep.wav(data_type='eval1'),
                'eval2': self.prep.wav(data_type='eval2'),
                'eval3': self.prep.wav(data_type='eval3'),
            }

        print('---------- train ----------')
        speaker_dict = read_sdb(label_paths=self.label_paths['train'],
                                run_root_path=self.prep.run_root_path,
                                model='ctc')
        train_global_mean_male, train_global_mean_female, train_global_std_male, train_global_std_female = read_audio(
            audio_paths=audio_paths['train'],
            tool=tool,
            config=None,
            speaker_dict=speaker_dict,
            normalize=normalize,
            is_training=True)

        for data_type in ['dev', 'eval1', 'eval2', 'eval3']:
            print('---------- %s ----------' % data_type)
            speaker_dict = read_sdb(label_paths=self.label_paths[data_type],
                                    run_root_path=self.prep.run_root_path,
                                    model='ctc')
            read_audio(audio_paths=audio_paths[data_type],
                       tool=tool,
                       config=None,
                       speaker_dict=speaker_dict,
                       normalize=normalize,
                       is_training=False,
                       train_global_mean_male=train_global_mean_male,
                       train_global_mean_female=train_global_mean_female,
                       train_global_std_male=train_global_std_male,
                       train_global_std_female=train_global_std_female)


if __name__ == '__main__':
    unittest.main()
