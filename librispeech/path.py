#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from os.path import join, basename, splitext
from glob import glob


class Path(object):
    """Prepare for making dataset.
    Args:
        data_path (string): path to Librispeech corpus
        htk_save_path (string): path to htk files
    """

    def __init__(self, data_path, htk_save_path=None):

        self.data_path = data_path
        self.htk_save_path = htk_save_path

        # Paths to Librispeech data
        self.data_paths = {
            'train-clean-100': join(data_path, 'train-clean-100'),
            'train_clean-360': join(data_path, 'train-clean-360'),
            'train-other-500': join(data_path, 'train-other-500'),
            'dev-clean': join(data_path, 'dev-clean'),
            'dev-other': join(data_path, 'dev-other'),
            'test-clean': join(data_path, 'test-clean'),
            'test-other': join(data_path, 'test-other')
        }

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

        self._wav_paths = {
            'train-clean-100': [],
            'train_clean-360': [],
            'train-other-500': [],
            'dev-clean': [],
            'dev-other': [],
            'test-clean': [],
            'test-other': []
        }

        self._trans_paths = {
            'train-clean-100': [],
            'train_clean-360': [],
            'train-other-500': [],
            'dev-clean': [],
            'dev-other': [],
            'test-clean': [],
            'test-other': []
        }

        for data_type in self._wav_paths.keys():
            for file_path in glob(join(self.data_paths[data_type], '*/*/*')):
                if splitext(basename(file_path))[1] == '.wav':
                    self._wav_paths[data_type].append(file_path)
                elif splitext(basename(file_path))[1] == '.txt':
                    self._trans_paths[data_type].append(file_path)

    def wav(self, data_type):
        """Get paths to wav files.
        Args:
            data_type (string): train100h or train460h or train960h or
                dev_clean or dev_other or test_clean or test_other
        Returns:
            list of paths to wav files
        """
        if data_type == 'train960h':
            return sorted(self._wav_paths['train-clean-100'] +
                          self._wav_paths['train_clean-360'] +
                          self._wav_paths['train-other-500'])
        elif data_type == 'train460h':
            return sorted(self._wav_paths['train-clean-100'] +
                          self._wav_paths['train_clean-360'])
        elif data_type == 'train100h':
            return sorted(self._wav_paths['train-clean-100'])
        else:
            return sorted(self._wav_paths[data_type.replace('_', '-')])

    def htk(self, data_type):
        """Get paths to htk files.
        Args:
            data_type (string): train100h or train460h or train960h or
                dev_clean or dev_other or test_clean or test_other
        Returns:
            list of paths to htk files
        """
        if self.htk_save_path is None:
            raise ValueError('Set path to htk files.')

        return [p.replace('wav', 'htk').replace(self.data_path, self.htk_save_path) for p in self.wav(data_type)]

    def trans(self, data_type):
        """Get paths to transcription files.
        Args:
            data_type (string): train100h or train460h or train960h or
                dev_clean or dev_other or test_clean or test_other
        Returns:
            list of paths to transcription files
        """
        if data_type == 'train960h':
            return sorted(self._trans_paths['train-clean-100'] +
                          self._trans_paths['train_clean-360'] +
                          self._trans_paths['train-other-500'])
        elif data_type == 'train460h':
            return sorted(self._trans_paths['train-clean-100'] +
                          self._trans_paths['train_clean-360'])
        elif data_type == 'train100h':
            return sorted(self._trans_paths['train-clean-100'])
        else:
            return sorted(self._trans_paths[data_type.replace('_', '-')])


if __name__ == '__main__':

    prep = Path(data_path='/n/sd8/inaguma/corpus/librispeech/data',
                htk_save_path='/n/sd8/inaguma/corpus/librispeech/htk')

    for data_type in ['train100h', 'train460h', 'train960h',
                      'dev_clean', 'dev_other', 'test_clean', 'test_other']:

        print('===== %s =====' % data_type)
        print(len(prep.wav(data_type=data_type)))
        print(len(prep.trans(data_type=data_type)))
        print(len(prep.htk(data_type=data_type)))
