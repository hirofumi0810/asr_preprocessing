#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset."""

from os.path import join, basename, isfile
import sys
from glob import glob

sys.path.append('../')
from utils.util import mkdir


class Prepare(object):
    """Prepare for making dataset."""

    def __init__(self):

        # path to csj data (set yourself)
        self.data_path = '/n/sd8/inaguma/corpus/csj/data/'
        self.wav_path = join(self.data_path, 'WAV')
        # update ver.
        self.ver4_path = join(self.data_path, 'Ver4/SDB')
        self.database_path = join(self.data_path, 'csj.db')

        # path to save directories (set yourself)
        self.dataset_path = mkdir('/n/sd8/inaguma/corpus/csj/dataset/')
        self.fbank_path = mkdir('/n/sd8/inaguma/corpus/csj/fbank/')

        # absolute path to this directory (set yourself)
        self.run_root_path = '/n/sd8/inaguma/src/asr/asr_preprocessing/src/csj/'

        self.__make()

    def __make(self):

        # read eval speaker list
        eval1_speakers, eval2_speakers, eval3_speakers = [], [], []
        excluded_speakers, dialog_dev_speakers, dialog_test_speakers = [], [], []
        # speakers in test data
        with open(join(self.run_root_path, 'eval1_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                eval1_speakers.append(speaker_name)
        with open(join(self.run_root_path, 'eval2_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                eval2_speakers.append(speaker_name)
        with open(join(self.run_root_path, 'eval3_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                eval3_speakers.append(speaker_name)

        # exclude speakers included in evaluation data
        with open(join(self.run_root_path, 'excluded_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                excluded_speakers.append(speaker_name)

        # speakers in dev dialog data
        # 自由対話のうち7対話（インタビュワー・インタビュイー両方含む）
        with open(join(self.run_root_path, 'dialog_dev_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                dialog_dev_speakers.append(speaker_name)

        # speakers in test dialog data
        # 自由対話のうち８対話（インタビュイーのみ）
        with open(join(self.run_root_path, 'dialog_test_speaker_list.txt'), 'r') as f:
            for line in f:
                speaker_name = line.strip()
                dialog_test_speakers.append(speaker_name)

        ####################
        # wav
        ####################
        self.wav_train_paths = []  # 967 A + 19 M
        self.wav_train_all_paths = []  # 3212 (A + S + M + R)
        self.wav_dev_paths = []
        self.wav_eval1_paths = []  # 10
        self.wav_eval2_paths = []  # 10
        self.wav_eval3_paths = []  # 10
        # dialog data
        # core: 30 (15 sessions)
        # noncore: 80 (40 sessions)
        # total => 110 (55 sessions)
        # exclude speakers included in exclude_speakersに含まれるものは除く
        # 102 (51 sessions)
        self.wav_dialog_train_paths = []  # 102-8=９４ files
        self.wav_dialog_dev_paths = []  # 7*2=１4 files
        self.wav_dialog_test_paths = []  # 8 files

        # core
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
                if speaker_name in dialog_dev_speakers:
                    self.wav_dialog_train_paths.append(wav_path)
                    self.wav_dialog_dev_paths.append(wav_path)
                elif speaker_name in dialog_test_speakers:
                    self.wav_dialog_test_paths.append(wav_path)
                elif speaker_name[-1] in ['L', 'R']:
                    self.wav_dialog_train_paths.append(wav_path)
            elif speaker_name[0] == 'A':
                self.wav_train_paths.append(wav_path)
                self.wav_train_all_paths.append(wav_path)
            else:
                self.wav_train_all_paths.append(wav_path)

        # noncore
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
                self.wav_train_all_paths.append(wav_path)
                if speaker_name[0] == 'M':
                    self.wav_dev_paths.append(wav_path)
            else:
                self.wav_train_all_paths.append(wav_path)

        # noncore dialog
        self.wav_noncore_dialog_paths = []
        for wav_path in glob(join(self.wav_path, 'NONCORE-DIALOG/*/*.wav')):
            speaker_name = basename(wav_path).split('.')[0]
            if speaker_name.split('-')[0] in excluded_speakers:
                continue
            elif speaker_name[0] == 'D':
                if speaker_name in dialog_dev_speakers:
                    self.wav_dialog_train_paths.append(wav_path)
                    self.wav_dialog_dev_paths.append(wav_path)
                elif speaker_name in dialog_test_speakers:
                    self.wav_dialog_test_paths.append(wav_path)
                elif speaker_name[-1] in ['L', 'R']:
                    self.wav_dialog_train_paths.append(wav_path)
            else:
                self.wav_train_all_paths.append(wav_path)

        ##################################
        # transcript (use ver4 if exists)
        ##################################
        self.trans_train_paths = []
        self.trans_train_all_paths = []
        self.trans_dev_paths = []
        self.trans_eval1_paths = []
        self.trans_eval2_paths = []
        self.trans_eval3_paths = []
        self.trans_dialog_train_paths = set([])
        self.trans_dialog_dev_paths = set([])
        self.trans_dialog_test_paths = set([])

        # train
        for index, wav_path in enumerate(self.wav_train_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_train_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_train_paths.append(ver4_path)
            else:
                self.trans_train_paths.append(wav_path.replace('.wav', '.sdb'))

        # train_all
        for index, wav_path in enumerate(self.wav_train_all_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_train_all_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_train_all_paths.append(ver4_path)
            else:
                self.trans_train_all_paths.append(
                    wav_path.replace('.wav', '.sdb'))

        # dev
        for index, wav_path in enumerate(self.wav_dev_paths):
            wav_path = join(self.wav_path, wav_path)
            self.wav_dev_paths[index] = wav_path
            self.trans_dev_paths.append(wav_path.replace('.wav', '.sdb'))

        # eval1
        for index, wav_path in enumerate(self.wav_eval1_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_eval1_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_eval1_paths.append(ver4_path)
            else:
                self.trans_eval1_paths.append(wav_path.replace('.wav', '.sdb'))

        # eval2
        for index, wav_path in enumerate(self.wav_eval2_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_eval2_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_eval2_paths.append(ver4_path)
            else:
                self.trans_eval2_paths.append(wav_path.replace('.wav', '.sdb'))

        # eval3
        for index, wav_path in enumerate(self.wav_eval3_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_eval3_paths[index] = wav_path
            ver4_path = join(self.ver4_path, speaker_name + '.sdb')
            if isfile(ver4_path):
                self.trans_eval3_paths.append(ver4_path)
            else:
                self.trans_eval3_paths.append(wav_path.replace('.wav', '.sdb'))

        # dialog train
        for index, wav_path in enumerate(self.wav_dialog_train_paths):
            speaker_name = basename(wav_path).split('.')[0]
            wav_path = join(self.wav_path, wav_path)
            self.wav_dialog_train_paths[index] = wav_path
            ver4_path = join(
                self.ver4_path, speaker_name.split('-')[0] + '.sdb')
            if isfile(ver4_path):
                self.trans_dialog_train_paths.add(ver4_path)
            else:
                self.trans_dialog_train_paths.add(wav_path.replace(
                    '-L.wav', '.sdb').replace('-R.wav', '.sdb'))
        self.trans_dialog_train_paths = list(self.trans_dialog_train_paths)

        # dialog dev
        for index, wav_path in enumerate(self.wav_dialog_dev_paths):
            wav_path = join(self.wav_path, wav_path)
            self.wav_dialog_dev_paths[index] = wav_path
            self.trans_dialog_dev_paths.add(wav_path.replace(
                '-L.wav', '.sdb').replace('-R.wav', '.sdb'))
        self.trans_dialog_dev_paths = list(self.trans_dialog_dev_paths)

        # dialog test
        for index, wav_path in enumerate(self.wav_dialog_test_paths):
            wav_path = join(self.wav_path, wav_path)
            self.wav_dialog_test_paths[index] = wav_path
            self.trans_dialog_test_paths.add(wav_path.replace(
                '-L.wav', '.sdb').replace('-R.wav', '.sdb'))
        self.trans_dialog_test_paths = list(self.trans_dialog_test_paths)

    def wav(self, data_type):
        """Get paths to wav files.
        Args:
            data_type: train or train_all or dev or eval1 or eval2 or eval3
            dialog_train or dialog_dev or dialog_test
        Returns:
            paths to wav files
        """
        if data_type == 'train':
            return sorted(self.wav_train_paths)
        elif data_type == 'train_all':
            return sorted(self.wav_train_all_paths)
        elif data_type == 'dev':
            return sorted(self.wav_dev_paths)
        elif data_type == 'eval1':
            return sorted(self.wav_eval1_paths)
        elif data_type == 'eval2':
            return sorted(self.wav_eval2_paths)
        elif data_type == 'eval3':
            return sorted(self.wav_eval3_paths)
        elif data_type == 'dialog_train':
            return sorted(self.wav_dialog_train_paths)
        elif data_type == 'dialog_dev':
            return sorted(self.wav_dialog_dev_paths)
        elif data_type == 'dialog_test':
            return sorted(self.wav_dialog_test_paths)

    def trans(self, data_type):
        """Get paths to transcription (.sdb) files.
        Args:
            data_type: train or train_all or dev or eval1 or eval2 or eval3
            dialog_train or dialog_dev or dialog_test
        Returns:
            paths to transcription files
        """
        if data_type == 'train':
            return sorted(self.trans_train_paths)
        elif data_type == 'train_all':
            return sorted(self.trans_train_all_paths)
        elif data_type == 'dev':
            return sorted(self.trans_dev_paths)
        elif data_type == 'eval1':
            return sorted(self.trans_eval1_paths)
        elif data_type == 'eval2':
            return sorted(self.trans_eval2_paths)
        elif data_type == 'eval3':
            return sorted(self.trans_eval3_paths)
        elif data_type == 'dialog_train':
            return sorted(self.trans_dialog_train_paths)
        elif data_type == 'dialog_dev':
            return sorted(self.trans_dialog_dev_paths)
        elif data_type == 'dialog_test':
            return sorted(self.trans_dialog_test_paths)


if __name__ == '__main__':
    prep = Prepare()

    print('===== train (240h) =====')
    print(len(prep.wav('train')))
    print(len(prep.trans('train')))

    print('===== train (586h) =====')
    print(len(prep.wav('train_all')))
    print(len(prep.trans('train_all')))

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

    print('===== dialog (train) =====')
    print(len(prep.wav('dialog_train')))
    print(len(prep.trans('dialog_train')))

    print('===== dialog (dev) =====')
    print(len(prep.wav('dialog_dev')))
    print(len(prep.trans('dialog_dev')))

    print('===== dialog (test) =====')
    print(len(prep.wav('dialog_test')))
    print(len(prep.trans('dialog_test')))
