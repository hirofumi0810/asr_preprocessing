#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from os.path import join, basename, splitext, abspath
from glob import glob


class Prepare(object):
    """Prepare for making dataset.
    Args:
        data_path (string): path to Librispeech corpus
        run_root_path (string): path to ./make.sh
    """

    def __init__(self, data_path, run_root_path):

        # Paths to Librispeech data
        self.data_paths = {
            'train_clean100': join(data_path, 'train-clean-100'),
            'train_clean360': join(data_path, 'train-clean-360'),
            'train_other500': join(data_path, 'train-other-500'),
            'dev_clean': join(data_path, 'dev-clean'),
            'dev_other': join(data_path, 'dev-other'),
            'test_clean': join(data_path, 'test-clean'),
            'test_other': join(data_path, 'test-other')
        }

        # Absolute path to this directory
        self.run_root_path = run_root_path

        # Load speaker data
        self.speaker_gender_dict = {}
        with open(join(data_path, 'SPEAKERS.TXT'), 'r') as f:
            for line in f:
                line = line.strip()
                if line[0] == ';':
                    continue
                # Remove consecutive spaces
                while '  ' in line:
                    line = re.sub(r'[\s]+', ' ', line)
                speaker = line.split(' ')[0]
                gender = line.split(' ')[2]
                self.speaker_gender_dict[speaker] = gender

        self.__make()

    def __make(self):

        self.wav_paths = {
            'train_clean100': [],
            'train_clean360': [],
            'train_other500': [],
            'dev_clean': [],
            'dev_other': [],
            'test_clean': [],
            'test_other': []
        }

        self.text_paths = {
            'train_clean100': [],
            'train_clean360': [],
            'train_other500': [],
            'dev_clean': [],
            'dev_other': [],
            'test_clean': [],
            'test_other': []
        }

        for data_type in self.wav_paths.keys():
            for file_path in glob(join(self.data_paths[data_type], '*/*/*')):
                if splitext(basename(file_path))[1] == '.wav':
                    self.wav_paths[data_type].append(file_path)
                elif splitext(basename(file_path))[1] == '.txt':
                    self.text_paths[data_type].append(file_path)

        file_number = {
            'train_clean100': 28539,
            'train_clean360': 104014,
            'train_other500': 148688,
            'dev_clean': 2703,
            'dev_other': 2864,
            'test_clean': 2620,
            'test_other': 2939
        }

        # for data_type in ['train_clean100', 'train_clean360', 'train_other500',
        #                   'dev_clean', 'dev_other', 'test_clean', 'test_other']:
        #     assert len(self.wav_paths[data_type]) == file_number[data_type], 'File number is not correct (True: %d, Now: %d).'.format(
        #         file_number[data_type], len(self.wav_paths[data_type]))

    def wav(self, data_type):
        """Get paths to wav files.
        Args:
            data_type (string): train_clean100 or train_clean360 or
                train_other500 or train_all or dev_clean or dev_other or
                test_clean or test_clean
        Returns:
            paths to wav files
        """
        if data_type == 'train_all':
            return sorted(self.wav_paths['train_clean100'] + self.wav_paths['train_clean360'] + self.wav_paths['train_other500'])
        return sorted(self.wav_paths[data_type])

    def text(self, data_type):
        """Get paths to transcription files.
        Args:
            data_type (string): train_clean100 or train_clean360 or
                train_other500 or train_all or dev_clean or dev_other or
                test_clean or test_clean
        Returns:
            paths to transcription files
        """
        if data_type == 'train_all':
            return sorted(self.text_paths['train_clean100'] + self.text_paths['train_clean360'] + self.text_paths['train_other500'])
        return sorted(self.text_paths[data_type])


if __name__ == '__main__':

    prep = Prepare(data_path='/n/sd8/inaguma/corpus/librispeech/data',
                   run_root_path=abspath('./'))

    for data_type in ['train_clean100', 'train_clean360', 'train_other500', 'train_all',
                      'dev_clean', 'dev_other', 'test_clean', 'test_other']:

        print('===== %s =====' % data_type)
        print(len(prep.wav(data_type=data_type)))
        print(len(prep.text(data_type=data_type)))
