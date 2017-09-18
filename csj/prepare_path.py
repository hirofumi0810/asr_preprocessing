#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from os.path import join, basename, isfile, abspath
import sys
from glob import glob

sys.path.append('../')
from utils.util import mkdir


class Prepare(object):
    """Prepare for making dataset.
    Args:
        data_path (string): path to csj corpus
        run_root_path (string): path to ./make.sh
        dataset_save_path (string): path to save dataset
    """

    def __init__(self, data_path, run_root_path):

        # Paths to CSJ data
        self.data_path = data_path
        self.wav_path = join(self.data_path, 'WAV')
        # NOTE: Update ver. (CSJ ver 4.)
        self.ver4_path = join(self.data_path, 'Ver4/SDB')

        # Absolute path to this directory
        self.run_root_path = run_root_path

        self.__make()

    def __make(self):

        # Load eval speaker list
        eval1_speakers, eval2_speakers, eval3_speakers = [], [], []
        excluded_speakers = []
        # Speakers in test data
        with open(join(self.run_root_path,
                       'config/eval1_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker = line.strip()
                eval1_speakers.append(speaker)
        with open(join(self.run_root_path,
                       'config/eval2_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker = line.strip()
                eval2_speakers.append(speaker)
        with open(join(self.run_root_path,
                       'config/eval3_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker = line.strip()
                eval3_speakers.append(speaker)

        # Exclude speakers in evaluation data
        with open(join(self.run_root_path,
                       'config/excluded_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker = line.strip()
                excluded_speakers.append(speaker)

        ####################
        # wav
        ####################
        self.wav_paths = {
            'train_subset': [],  # 967 A + 19 M files
            'train_fullset': [],  # 3212 (A + S + M + R) files
            'dev': [],   # 19 files
            'eval1': [],  # 10 files
            'eval2': [],  # 10 files
            'eval3': [],  # 10 files
            'dialog': []
        }

        # Core
        for wav_path in glob(join(self.wav_path, 'CORE/*/*/*.wav')):
            speaker = basename(wav_path).split('.')[0]
            if speaker in eval1_speakers:
                self.wav_paths['eval1'].append(wav_path)
            elif speaker in eval2_speakers:
                self.wav_paths['eval2'].append(wav_path)
            elif speaker in eval3_speakers:
                self.wav_paths['eval3'].append(wav_path)
            elif speaker.split('-')[0] in excluded_speakers:
                continue
            elif speaker[0] == 'D':
                if speaker not in excluded_speakers and '-R' in speaker and 'D03' in speaker:
                    self.wav_paths['dialog'].append(wav_path)
                continue
            elif speaker[0] == 'A':
                self.wav_paths['train_subset'].append(wav_path)
                self.wav_paths['train_fullset'].append(wav_path)
            else:
                self.wav_paths['train_fullset'].append(wav_path)

        # Noncore
        for wav_path in glob(join(self.wav_path, 'NONCORE/*/*/*/*.wav')):
            speaker = basename(wav_path).split('.')[0]
            if speaker in eval1_speakers:
                self.wav_paths['eval1'].append(wav_path)
            elif speaker in eval2_speakers:
                self.wav_paths['eval2'].append(wav_path)
            elif speaker in eval3_speakers:
                self.wav_paths['eval3'].append(wav_path)
            elif speaker.split('-')[0] in excluded_speakers:
                continue
            elif speaker[0] in ['A', 'M']:
                self.wav_paths['train_subset'].append(wav_path)
                self.wav_paths['train_fullset'].append(wav_path)
                if speaker[0] == 'M':
                    self.wav_paths['dev'].append(wav_path)
            else:
                self.wav_paths['train_fullset'].append(wav_path)

        # Noncore dialog
        for wav_path in glob(join(self.wav_path, 'NONCORE-DIALOG/*/*.wav')):
            speaker = basename(wav_path).split('.')[0]
            if speaker.split('-')[0] in excluded_speakers:
                continue
            elif speaker[0] == 'D':
                if speaker not in excluded_speakers and '-R' in speaker and 'D03' in speaker:
                    self.wav_paths['dialog'].append(wav_path)
                continue
            else:
                self.wav_paths['train_fullset'].append(wav_path)

        ##################################
        # Transcript (use ver4 if exists)
        ##################################
        self.trans_paths = {
            'train_subset': [],
            'train_fullset': [],
            'dev': [],
            'eval1': [],
            'eval2': [],
            'eval3': [],
            'dialog': []
        }

        # train subset (about 240h)
        for i, wav_path in enumerate(self.wav_paths['train_subset']):
            speaker = basename(wav_path).split('.')[0]
            self.wav_paths['train_subset'][i] = wav_path
            ver4_path = join(self.ver4_path, speaker + '.sdb')
            if isfile(ver4_path):
                self.trans_paths['train_subset'].append(ver4_path)
            else:
                self.trans_paths['train_subset'].append(wav_path.replace('.wav', '.sdb'))

        # train fullset (about 586h)
        for i, wav_path in enumerate(self.wav_paths['train_fullset']):
            speaker = basename(wav_path).split('.')[0]
            self.wav_paths['train_fullset'][i] = wav_path
            ver4_path = join(self.ver4_path, speaker + '.sdb')
            if isfile(ver4_path):
                self.trans_paths['train_fullset'].append(ver4_path)
            else:
                self.trans_paths['train_fullset'].append(wav_path.replace('.wav', '.sdb'))

        # Dev
        for i, wav_path in enumerate(self.wav_paths['dev']):
            self.wav_paths['dev'][i] = wav_path
            self.trans_paths['dev'].append(wav_path.replace('.wav', '.sdb'))

        # eval1, eval2, evak3
        for data_type in ['eval1', 'eval2', 'eval3']:
            for i, wav_path in enumerate(self.wav_paths[data_type]):
                speaker = basename(wav_path).split('.')[0]
                wav_path = join(self.wav_path, wav_path)
                self.wav_paths[data_type][i] = wav_path
                ver4_path = join(self.ver4_path, speaker + '.sdb')
                if isfile(ver4_path):
                    self.trans_paths[data_type].append(ver4_path)
                else:
                    self.trans_paths[data_type].append(wav_path.replace('.wav', '.sdb'))

        # Dialog
        for i, wav_path in enumerate(self.wav_paths['dialog']):
            wav_path = re.sub('-R', '', wav_path)
            speaker = basename(wav_path).split('.')[0]
            self.wav_paths['dialog'][i] = wav_path
            ver4_path = join(self.ver4_path, speaker + '.sdb')
            if isfile(ver4_path):
                self.trans_paths['dialog'].append(ver4_path)
            else:
                self.trans_paths['dialog'].append(wav_path.replace('.wav', '.sdb'))

        file_number = {
            'train_subset': 986,
            'train_fullset': 3212,
            'dev': 19,
            'eval1': 10,
            'eval2': 10,
            'eval3': 10
        }

        for data_type in ['train_fullset', 'train_subset', 'dev', 'eval1', 'eval2', 'eval3']:
            assert len(self.wav_paths[data_type]) == file_number[data_type], 'File number is not correct (True: %d, Now: %d).'.format(
                file_number[data_type], len(self.wav_paths[data_type]))

    def wav(self, data_type):
        """Get paths to wav files.
        Args:
            data_type (string): train_subset or train_fullset or dev or eval1 or
                eval2 or eval3 or dialog
        Returns:
            paths to wav files
        """
        return sorted(self.wav_paths[data_type])

    def trans(self, data_type):
        """Get paths to transcription (.sdb) files.
        Args:
            data_type (string): train_subset or train_fullset or dev or eval1 or
                eval2 or eval3 or dialog
        Returns:
            paths to transcription files
        """
        return sorted(self.trans_paths[data_type])


if __name__ == '__main__':

    data_path = '/n/sd8/inaguma/corpus/csj/data'

    prep = Prepare(data_path, abspath('./'))

    print('===== train subset (240h) =====')
    print(len(prep.wav('train_subset')))
    print(len(prep.trans('train_subset')))

    print('===== train fullset (586h) =====')
    print(len(prep.wav('train_fullset')))
    print(len(prep.trans('train_fullset')))

    print('===== dev =====')
    print(len(prep.wav('dev')))
    print(len(prep.trans('dev')))

    print('===== eval1 =====')
    print(len(prep.wav('eval1')))
    print(len(prep.trans('eval1')))

    print('===== eval2 =====')
    print(len(prep.wav('eval2')))
    print(len(prep.trans('eval2')))

    print('===== eval3 =====')
    print(len(prep.wav('eval3')))
    print(len(prep.trans('eval3')))

    print('===== dialog =====')
    print(len(prep.wav('dialog')))
    print(len(prep.trans('dialog')))
