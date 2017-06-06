#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset."""

import os
from os.path import join
import sys
from glob import glob

sys.path.append('../')
from utils.util import mkdir_join


class Prepare(object):
    """Prepare for making dataset."""

    def __init__(self):

        # path to switchboard data (set yourself)
        self.data_path = '/n/sd8/inaguma/corpus/switchboard/'
        self.train_data_path = join(self.data_path, 'ldc97s62')
        self.train_data_fisher_path = join(self.data_path, 'fisher')
        self.test_data_path = join(
            self.data_path, 'eval2000')  # swbd + callhome

        # path to save directories (set yourself)
        self.dataset_path = mkdir_join(self.data_path, 'dataset')

        # absolute path to this directory (set yourself)
        self.run_root_path = '/n/sd8/inaguma/src/asr/asr_preprocessing/src/switchboard/'

        self.__make()

    def __make(self):

        ####################
        # train (ldc97s62)
        ####################
        self.sph_train_paths = []
        for sph_path in glob(join(self.train_data_path, '*/*/*/*.sph')):
            self.sph_train_paths.append(join(self.train_data_path, sph_path))

        self.pronounce_dict_path = join(
            self.train_data_path, 'ldc97s62/transcriptions/swb_ms98_transcriptions/sw-ms98-dict.text')
        self.word_train_paths = []
        self.trans_train_paths = []
        for text_path in glob(join(self.train_data_path,
                                   'ldc97s62/transcriptions/swb_ms98_transcriptions/*/*/*.text')):
            if text_path.split('.')[0][-4:] == 'word':
                self.word_train_paths.append(join(self.train_data_path,
                                                  text_path))
            elif text_path.split('.')[0][-5:] == 'trans':
                self.trans_train_paths.append(join(self.train_data_path,
                                                   text_path))

        ####################
        # train (fisher)
        ####################
        self.sph_train_fisher_paths = []
        for sph_path in glob(join(self.train_data_fisher_path,
                                  'fisher_english/audio/*/*.sph')):
            self.sph_train_fisher_paths.append(
                join(self.train_data_fisher_path, sph_path))

        self.trans_train_fisher_paths = []
        for text_name in glob(join(self.train_data_fisher_path,
                                   'fisher_english/data/trans/*/*.txt')):
            self.trans_train_fisher_paths.append(
                join(self.train_data_fisher_path, text_name))

        ########################################
        # test (eval2000 [swbd + callhome])
        ########################################
        self.sph_test_paths = []
        self.sph_test_callhome_paths = []
        for file_path in glob(join(self.test_data_path,
                                   'ldc2002s09/english/*')):
            file_name = os.path.basename(file_path)
            if file_name == 'hub5e_00.pem':
                # hub5e_00.pem file is a segmentation file
                pass
            elif file_name[:2] == 'sw':
                self.sph_test_paths.append(
                    join(self.test_data_path, file_path))
            elif file_name[:2] == 'en':
                self.sph_test_callhome_paths.append(
                    join(self.test_data_path, file_path))

        self.trans_test_swbd_paths = []
        self.trans_test_callhome_paths = []
        for file_path in glob(join(self.test_data_path,
                                   'ldc2002t43/reference/english/*')):
            file_name = os.path.basename(file_path)
            # Switchboard evaluation transcript
            if file_name[:2] == 'sw':
                self.trans_test_swbd_paths.append(
                    join(self.test_data_path, file_path))
            # CallHome evaluation transcript
            elif file_name[:2] == 'en':
                self.trans_test_callhome_paths.append(
                    join(self.test_data_path, file_path))

    def sph_train(self, train_type='ldc97s62'):
        """Get paths to sph files of training data.
        Args:
            train_type: ldc97s62 or fisher
        Returns:
            paths to sph files
        """
        if train_type == 'ldc97s62':
            return sorted(self.sph_train_paths)
        elif train_type == 'fisher':
            return sorted(self.sph_train_fisher_paths)

    def sph_test(self, test_type='swbd'):
        """Get paths to sph files of evaluation data (eval2000).
        Args:
            test_type: swbd or callhome
        Returns:
            paths to sph files
        """
        if test_type == 'swbd':
            return sorted(self.sph_test_paths)
        elif test_type == 'callhome':
            return sorted(self.sph_test_callhome_paths)

    def label_train(self, label_type, train_type='ldc97s62'):
        """Get paths to transcription files of training data.
        Args:
            label_type: phone or character or word
            train_type: ldc97s62 or fisher
        Returns:
            paths: paths to transcription files
        """
        if train_type == 'ldc97s62':
            if label_type in ['character', 'phone']:
                return sorted(self.trans_train_paths)
            elif label_type == 'word':
                return sorted(self.word_train_paths)
        elif train_type == 'fisher':
            return sorted(self.trans_train_fisher_paths)

    def label_test(self, test_type='swbd'):
        """Get paths to transcription files of evaluation data (eval200).
        Args:
            test_type: swbd or callhome
        Returns:
            paths: paths to transcription files
        """
        if test_type == 'swbd':
            return sorted(self.trans_test_swbd_paths)
        elif test_type == 'callhome':
            return sorted(self.trans_test_callhome_paths)


if __name__ == '__main__':
    prep = Prepare()

    print('===== ldc97s62 ====')
    print(len(prep.sph_train()))
    print(len(prep.label_train(label_type='word')))
    print(len(prep.label_train(label_type='character')))

    print('==== fisher ====')
    print(len(prep.sph_train('fisher')))
    print(len(prep.label_train(label_type='character', train_type='fisher')))

    print('==== eval2000 (swbd) ====')
    print(len(prep.sph_test()))
    print(len(prep.label_test()))

    print('==== eval2000 (callhome) ====')
    print(len(prep.sph_test('callhome')))
    print(len(prep.label_test('callhome')))
