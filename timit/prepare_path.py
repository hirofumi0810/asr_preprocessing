#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename, splitext, abspath
from glob import glob


class Prepare(object):
    """Prepare for making dataset.
    Args:
        data_path (string): path to timit corpus
        run_root_path (string): path to ./make.sh
        dataset_save_path (string): path to save dataset
    """

    def __init__(self, data_path, run_root_path):

        # Paths to TIMIT data
        self.train_data_path = join(data_path, 'train')
        self.test_data_path = join(data_path, 'test')

        # Absolute path to this directory
        self.run_root_path = run_root_path

        self.__make()

    def __make(self):

        self.wav_paths = {}
        self.text_paths = {}
        self.word_paths = {}
        self.phone_paths = {}

        for data_type in ['train', 'dev', 'test']:

            self.wav_paths[data_type] = []
            self.text_paths[data_type] = []
            self.word_paths[data_type] = []
            self.phone_paths[data_type] = []
            data_path = self.train_data_path if data_type == 'train' else self.test_data_path

            if data_type != 'train':
                # Load speaker list
                test_speakers = []
                with open(join(self.run_root_path,
                               'config/' + data_type + '_speaker_list.txt'), 'r') as f:
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
                        self.wav_paths[data_type].append(join(data_path, file_path))
                    elif ext == '.txt':
                        self.text_paths[data_type].append(join(data_path, file_path))
                    elif ext == '.wrd':
                        self.word_paths[data_type].append(join(data_path, file_path))
                    elif ext == '.phn':
                        self.phone_paths[data_type].append(join(data_path, file_path))

        file_number = {
            'train': 3696,
            'dev': 400,
            'test': 192
        }

        for data_type in ['train', 'dev', 'test']:
            assert len(self.wav_paths[data_type]) == file_number[data_type], 'File number is not correct (True: %d, Now: %d).'.format(
                file_number[data_type], len(self.wav_paths[data_type]))

    def wav(self, data_type):
        """Get paths to wav files.
        Args:
            data_type (string): train or dev or test
        Returns:
            paths to wav files
        """
        return sorted(self.wav_paths[data_type])

    def text(self, data_type):
        """Get paths to sentence-level transcription files.
        Args:
            data_type (string): train or dev or test
        Returns:
            paths to transcription files
        """
        return sorted(self.text_paths[data_type])

    def word(self, data_type):
        """Get paths to word-level transcription files.
        Args:
            data_type (string): train or dev or test
        Returns:
            paths to transcription files
        """
        return sorted(self.word_paths[data_type])

    def phone(self, data_type):
        """Get paths to phone-level transcription files.
        Args:
            data_type (string): train or dev or test
        Returns:
            paths to transcription files
        """
        return sorted(self.phone_paths[data_type])


if __name__ == '__main__':

    prep = Prepare(data_path='/n/sd8/inaguma/corpus/timit/original',
                   run_root_path=abspath('./'))

    print('===== train =====')
    print(len(prep.wav(data_type='train')))
    print(len(prep.text(data_type='train')))
    print(len(prep.word(data_type='train')))
    print(len(prep.phone(data_type='train')))

    print('===== dev ======')
    print(len(prep.wav(data_type='dev')))
    print(len(prep.text(data_type='dev')))
    print(len(prep.word(data_type='dev')))
    print(len(prep.phone(data_type='dev')))

    print('===== test =====')
    print(len(prep.wav(data_type='test')))
    print(len(prep.text(data_type='test')))
    print(len(prep.word(data_type='test')))
    print(len(prep.phone(data_type='test')))
