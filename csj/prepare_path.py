#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename, isfile, abspath
import sys
from glob import glob

sys.path.append('../')
from utils.util import mkdir


class Prepare(object):
    """Prepare for making dataset.
    Args:
        csj_path: path to csj corpus
        run_root_path: path to ./make.sh
        dataset_save_path: path to save dataset
    """

    def __init__(self, csj_path, run_root_path):

        # Path to timit data
        self.data_path = csj_path
        self.wav_path = join(self.data_path, 'WAV')
        # update ver.
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
                speaker_name = line.strip()
                eval1_speakers.append(speaker_name)
        with open(join(self.run_root_path,
                       'config/eval2_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                eval2_speakers.append(speaker_name)
        with open(join(self.run_root_path,
                       'config/eval3_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                eval3_speakers.append(speaker_name)

        # Exclude speakers in evaluation data
        with open(join(self.run_root_path,
                       'config/excluded_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                excluded_speakers.append(speaker_name)

        ####################
        # wav
        ####################
        # Monolog
        self.wav_train_paths = []  # 967 A + 19 M files
        self.wav_train_large_paths = []  # 3212 (A + S + M + R) files
        self.wav_dev_paths = []  # 19 files
        self.wav_eval1_paths = []  # 10 files
        self.wav_eval2_paths = []  # 10 files
        self.wav_eval3_paths = []  # 10 files

        # Core
        for wav_path in glob(join(self.wav_path, 'CORE/*/*/*.wav')):
            speaker_name = basename(wav_path).split('.')[0]
            if speaker_name in eval1_speakers:
                self.wav_eval1_paths.append(wav_path)
            elif speaker_name in eval2_speakers:
                self.wav_eval2_paths.append(wav_path)
            elif speaker_name in eval3_speakers:
                self.wav_eval3_paths.append(wav_path)
            elif speaker_name.split('-')[0] in excluded_speakers:
                continue
            elif speaker_name[0] == 'D':
                continue
            elif speaker_name[0] == 'A':
                self.wav_train_paths.append(wav_path)
                self.wav_train_large_paths.append(wav_path)
            else:
                self.wav_train_large_paths.append(wav_path)

        # Noncore
        for wav_path in glob(join(self.wav_path, 'NONCORE/*/*/*/*.wav')):
            speaker_name = basename(wav_path).split('.')[0]
            if speaker_name in eval1_speakers:
                self.wav_eval1_paths.append(wav_path)
            elif speaker_name in eval2_speakers:
                self.wav_eval2_paths.append(wav_path)
            elif speaker_name in eval3_speakers:
                self.wav_eval3_paths.append(wav_path)
            elif speaker_name.split('-')[0] in excluded_speakers:
                continue
            elif speaker_name[0] in ['A', 'M']:
                self.wav_train_paths.append(wav_path)
                self.wav_train_large_paths.append(wav_path)
                if speaker_name[0] == 'M':
                    self.wav_dev_paths.append(wav_path)
            else:
                self.wav_train_large_paths.append(wav_path)

        # Noncore dialog
        self.wav_noncore_dialog_paths = []
        for wav_path in glob(join(self.wav_path, 'NONCORE-DIALOG/*/*.wav')):
            speaker_name = basename(wav_path).split('.')[0]
            if speaker_name.split('-')[0] in excluded_speakers:
                continue
            elif speaker_name[0] == 'D':
                continue
            else:
                self.wav_train_large_paths.append(wav_path)

        ##################################
        # Transcript (use ver4 if exists)
        ##################################
        self.trans_train_paths = []
        self.trans_train_large_paths = []
        self.trans_dev_paths = []
        self.trans_eval1_paths = []
        self.trans_eval2_paths = []
        self.trans_eval3_paths = []

        # Train (about 240h)
        for index, wav_path in enumerate(self.wav_train_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_train_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_train_paths.append(ver4_path)
            else:
                self.trans_train_paths.append(wav_path.replace('.wav', '.sdb'))

        # Train_large (about 586h)
        for index, wav_path in enumerate(self.wav_train_large_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_train_large_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_train_large_paths.append(ver4_path)
            else:
                self.trans_train_large_paths.append(
                    wav_path.replace('.wav', '.sdb'))

        # Dev
        for index, wav_path in enumerate(self.wav_dev_paths):
            wav_path = join(self.wav_path, wav_path)
            self.wav_dev_paths[index] = wav_path
            self.trans_dev_paths.append(wav_path.replace('.wav', '.sdb'))

        # Eval1
        for index, wav_path in enumerate(self.wav_eval1_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_eval1_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_eval1_paths.append(ver4_path)
            else:
                self.trans_eval1_paths.append(wav_path.replace('.wav', '.sdb'))

        # Eval2
        for index, wav_path in enumerate(self.wav_eval2_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_eval2_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_eval2_paths.append(ver4_path)
            else:
                self.trans_eval2_paths.append(wav_path.replace('.wav', '.sdb'))

        # Eval3
        for index, wav_path in enumerate(self.wav_eval3_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_eval3_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_eval3_paths.append(ver4_path)
            else:
                self.trans_eval3_paths.append(wav_path.replace('.wav', '.sdb'))

    def wav(self, data_type):
        """Get paths to wav files.
        Args:
            data_type: string, train or train_large or dev or eval1 or eval2,
                eval3
        Returns:
            paths to wav files
        """
        if data_type == 'train':
            return sorted(self.wav_train_paths)
        elif data_type == 'train_large':
            return sorted(self.wav_train_large_paths)
        elif data_type == 'dev':
            return sorted(self.wav_dev_paths)
        elif data_type == 'eval1':
            return sorted(self.wav_eval1_paths)
        elif data_type == 'eval2':
            return sorted(self.wav_eval2_paths)
        elif data_type == 'eval3':
            return sorted(self.wav_eval3_paths)

    def trans(self, data_type):
        """Get paths to transcription (.sdb) files.
        Args:
            data_type: string, train or train_large or dev or eval1 or eval2,
                eval3
        Returns:
            paths to transcription files
        """
        if data_type == 'train':
            return sorted(self.trans_train_paths)
        elif data_type == 'train_large':
            return sorted(self.trans_train_large_paths)
        elif data_type == 'dev':
            return sorted(self.trans_dev_paths)
        elif data_type == 'eval1':
            return sorted(self.trans_eval1_paths)
        elif data_type == 'eval2':
            return sorted(self.trans_eval2_paths)
        elif data_type == 'eval3':
            return sorted(self.trans_eval3_paths)


if __name__ == '__main__':

    csj_path = '/n/sd8/inaguma/corpus/csj/data/'

    prep = Prepare(csj_path, abspath('./'))

    print('===== train (240h) =====')
    print(len(prep.wav('train')))
    print(len(prep.trans('train')))

    print('===== train_large (586h) =====')
    print(len(prep.wav('train_large')))
    print(len(prep.trans('train_large')))

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
