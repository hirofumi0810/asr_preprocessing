#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset."""

import os
import sys
import glob

sys.path.append('../')
from utils.util import mkdir


class Prepare(object):
    """Prepare for making dataset."""

    def __init__(self):

        # path to csj data (set yourself)
        self.data_path = '/n/sd8/inaguma/corpus/csj/data/'
        self.wav_path = os.path.join(self.data_path, 'WAV')
        self.trans_path = os.path.join(self.data_path, 'Ver4')

        # path to save directories (set yourself)
        self.dataset_path = mkdir('/n/sd8/inaguma/corpus/csj/dataset/')
        self.fbank_path = mkdir('/n/sd8/inaguma/corpus/csj/fbank/')

        # absolute path to this directory (set yourself)
        self.run_root_path = '/n/sd8/inaguma/src/asr/asr_preprocessing/src/csj/'

        self.__make()

    def __make(self):

        # read eval speaker list
        eval1, eval2, eval3, excluded = [], [], [], []
        with open(os.path.join(self.run_root_path, 'eval1_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                eval1.append(speaker_name)
        with open(os.path.join(self.run_root_path, 'eval2_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                eval2.append(speaker_name)
        with open(os.path.join(self.run_root_path, 'eval3_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                eval3.append(speaker_name)

        # exclude speakers included in evaluation data
        with open(os.path.join(self.run_root_path, 'excluded_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                excluded.append(speaker_name)

        ####################
        # wav
        ####################
        self.wav_train_paths = []  # 967 A + 19 M
        # all except for dialog (only diffrence) => as results, 3212
        self.wav_train_plus_paths = []
        self.wav_dev_paths = []
        self.wav_eval1_paths = []  # 10
        self.wav_eval2_paths = []  # 10
        self.wav_eval3_paths = []  # 10
        self.wav_dialog_paths = []  # 165

        # core
        for wav_path in glob.glob(os.path.join(self.wav_path, 'CORE/*/*/*.wav')):
            speaker_name = wav_path.split('/')[-2]
            if speaker_name in eval1:
                self.wav_eval1_paths.append(wav_path)
            elif speaker_name in eval2:
                self.wav_eval2_paths.append(wav_path)
            elif speaker_name in eval3:
                self.wav_eval3_paths.append(wav_path)
            elif speaker_name in excluded:
                continue
            elif speaker_name[0] == 'D':
                self.wav_dialog_paths.append(wav_path)
            elif speaker_name[0] == 'A':
                self.wav_train_paths.append(wav_path)
            else:
                self.wav_train_plus_paths.append(wav_path)

        # non core
        for wav_path in glob.glob(os.path.join(self.wav_path, 'NONCORE/*/*/*/*.wav')):
            speaker_name = wav_path.split('/')[-2]
            if speaker_name in eval1:
                self.wav_eval1_paths.append(wav_path)
            elif speaker_name in eval2:
                self.wav_eval2_paths.append(wav_path)
            elif speaker_name in eval3:
                self.wav_eval3_paths.append(wav_path)
            elif speaker_name in excluded:
                continue
            elif speaker_name[0] in ['A', 'M']:
                self.wav_train_paths.append(wav_path)
            else:
                self.wav_train_plus_paths.append(wav_path)

        # non core dialog
        self.wav_noncore_dialog_paths = []
        for wav_path in glob.glob(os.path.join(self.wav_path, 'NONCORE-DIALOG/*/*.wav')):
            speaker_name = wav_path.split('/')[-2]
            if speaker_name[0] == 'D':
                self.wav_dialog_paths.append(wav_path)
            else:
                self.wav_train_plus_paths.append(wav_path)

        ####################
        # transcript
        ####################
        self.trans_train_paths = []
        self.trans_train_plus_paths = []
        self.trans_dev_paths = []
        self.trans_eval1_paths = []
        self.trans_eval2_paths = []
        self.trans_eval3_paths = []
        self.trans_dialog_paths = []

        for index, wav_path in enumerate(self.wav_train_paths):
            self.wav_train_paths[index] = os.path.join(self.wav_path, wav_path)
            trans_path = wav_path.replace('.wav', '.sdb')
            self.trans_train_paths.append(
                os.path.join(self.trans_path, trans_path))

        for index, wav_path in enumerate(self.wav_train_plus_paths):
            self.wav_train_plus_paths[index] = os.path.join(
                self.wav_path, wav_path)
            trans_path = wav_path.replace('.wav', '.sdb')
            self.trans_train_plus_paths.append(
                os.path.join(self.trans_path, trans_path))

        for index, wav_path in enumerate(self.wav_eval1_paths):
            self.wav_eval1_paths[index] = os.path.join(self.wav_path, wav_path)
            trans_path = wav_path.replace('.wav', '.sdb')
            self.trans_eval1_paths.append(
                os.path.join(self.trans_path, trans_path))

        for index, wav_path in enumerate(self.wav_eval2_paths):
            self.wav_eval2_paths[index] = os.path.join(self.wav_path, wav_path)
            trans_path = wav_path.replace('.wav', '.sdb')
            self.trans_eval2_paths.append(
                os.path.join(self.trans_path, trans_path))

        for index, wav_path in enumerate(self.wav_eval3_paths):
            self.wav_eval3_paths[index] = os.path.join(self.wav_path, wav_path)
            trans_path = wav_path.replace('.wav', '.sdb')
            self.trans_eval3_paths.append(
                os.path.join(self.trans_path, trans_path))

        for index, wav_path in enumerate(self.wav_dialog_paths):
            self.wav_dialog_paths[index] = os.path.join(
                self.wav_path, wav_path)
            trans_path = wav_path.replace('.wav', '.sdb')
            self.trans_dialog_paths.append(
                os.path.join(self.trans_path, trans_path))

        # update for ver.4 (core data only)

    def wav(self, data_type):
        """Get paths to wav files.
        Args:
            data_type: train or train_plus or dev or eval{1, 2, 3} or dialog
        Returns:
            paths to wav files
        """
        if data_type == 'train':
            return sorted(self.wav_train_paths)
        elif data_type == 'train_plus':
            return sorted(self.wav_train_plus_paths)
        elif data_type == 'dev':
            return sorted(self.wav_dev_paths)
        elif data_type == 'eval1':
            return sorted(self.wav_eval1_paths)
        elif data_type == 'eval2':
            return sorted(self.wav_eval2_paths)
        elif data_type == 'eval3':
            return sorted(self.wav_eval3_paths)
        elif data_type == 'dialog':
            return sorted(self.wav_dialog_paths)

    def trans(self, data_type):
        """Get paths to transcription (.sdb) files.
        Args:
            data_type: train or train_plus or dev or eval{1, 2, 3} or dialog
        Returns:
            paths to transcription files
        """
        if data_type == 'train':
            return sorted(self.trans_train_paths)
        elif data_type == 'train_plus':
            return sorted(self.trans_train_plus_paths)
        elif data_type == 'dev':
            return sorted(self.trans_dev_paths)
        elif data_type == 'eval1':
            return sorted(self.trans_eval1_paths)
        elif data_type == 'eval2':
            return sorted(self.trans_eval2_paths)
        elif data_type == 'eval3':
            return sorted(self.trans_eval3_paths)
        elif data_type == 'dialog':
            return sorted(self.trans_dialog_paths)


if __name__ == '__main__':
    prep = Prepare()

    print('===== train =====')
    print(len(prep.wav('train')))
    print(len(prep.trans('train')))

    print('===== train (all) =====')
    print(len(prep.wav('train') + prep.wav('train_plus')))
    print(len(prep.trans('train') + prep.trans('train_plus')))

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
