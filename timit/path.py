#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename, splitext
from glob import glob


class Path(object):
    """Prepare for making dataset.
    Args:
        data_path (string): path to TIMIT corpus
        config_path (string): path to config dir
        htk_save_path (string, optional): path to htk files
    """

    def __init__(self, data_path, config_path, htk_save_path=None):

        self.data_path = data_path
        self.config_path = config_path
        self.htk_save_path = htk_save_path

        # Paths to TIMIT data
        self.train_data_path = join(data_path, 'train')
        self.test_data_path = join(data_path, 'test')

        self.__make()

    def __make(self):

        self._wav_paths = {}
        self._text_paths = {}
        self._word_paths = {}
        self._phone_paths = {}

        for data_type in ['train', 'dev', 'test']:

            self._wav_paths[data_type] = []
            self._text_paths[data_type] = []
            self._word_paths[data_type] = []
            self._phone_paths[data_type] = []
            data_path = self.train_data_path if data_type == 'train' else self.test_data_path

            if data_type != 'train':
                # Load speaker list
                test_speakers = []
                with open(join(self.config_path, data_type + '_speaker_list.txt'), 'r') as f:
                    for line in f:
                        line = line.strip()
                        test_speakers.append(line)

            for file_path in glob(join(data_path, '*/*/*')):
                region, speaker, file_name = file_path.split('/')[-3:]
                ext = splitext(file_name)[1]

                if data_type != 'train' and speaker not in test_speakers:
                    continue

                if basename(file_name)[0: 2] in ['sx', 'si']:
                    if ext == '.wav':
                        self._wav_paths[data_type].append(
                            join(data_path, file_path))
                    elif ext == '.txt':
                        self._text_paths[data_type].append(
                            join(data_path, file_path))
                    elif ext == '.wrd':
                        self._word_paths[data_type].append(
                            join(data_path, file_path))
                    elif ext == '.phn':
                        self._phone_paths[data_type].append(
                            join(data_path, file_path))

    def wav(self, data_type):
        """Get paths to wav files.
        Args:
            data_type (string): train or dev or test
        Returns:
            list of paths to wav files
        """
        return sorted(self._wav_paths[data_type])

    def htk(self, data_type):
        """Get paths to htk files.
        Args:
            data_type (string): train or dev or test
        Returns:
            list of paths to htk files
        """
        if self.htk_save_path is None:
            raise ValueError('Set path to htk files.')

        # NOTE: ex.) timit/htk/data_type/speaker/*.htk
        return [p for p in glob(join(self.htk_save_path, data_type, '*/*.htk'))]

    def trans(self, data_type):
        """Get paths to sentence-level transcription files.
        Args:
            data_type (string): train or dev or test
        Returns:
            list of paths to transcription files
        """
        return sorted(self._text_paths[data_type])

    def word(self, data_type):
        """Get paths to word-level transcription files.
        Args:
            data_type (string): train or dev or test
        Returns:
            list of paths to transcription files
        """
        return sorted(self._word_paths[data_type])

    def phone(self, data_type):
        """Get paths to phone-level transcription files.
        Args:
            data_type (string): train or dev or test
        Returns:
            list of paths to transcription files
        """
        return sorted(self._phone_paths[data_type])


if __name__ == '__main__':

    path = Path(data_path='/n/sd8/inaguma/corpus/timit/data',
                config_path='./config',
                htk_save_path='/n/sd8/inaguma/corpus/timit/htk')

    for data_type in ['train', 'dev', 'test']:

        print('===== %s ======' % data_type)
        print(len(path.wav(data_type=data_type)))
        print(len(path.htk(data_type=data_type)))
        print(len(path.trans(data_type=data_type)))
        print(len(path.word(data_type=data_type)))
        print(len(path.phone(data_type=data_type)))
