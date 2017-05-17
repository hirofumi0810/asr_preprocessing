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

        # path to switchboard data (set yourself)
        self.data_path = '/n/sd8/inaguma/corpus/switchboard/'
        self.train_data_path = os.path.join(self.data_path, 'ldc97s62')
        self.train_data_fisher_path = os.path.join(self.data_path, 'fisher')
        self.test_data_path = os.path.join(
            self.data_path, 'eval2000')  # swbd + callhome

        # path to save directories (set yourself)
        self.dataset_path = mkdir(os.path.join(self.data_path, 'dataset'))

        # absolute path to this directory (set yourself)
        self.run_root_path = '/n/sd8/inaguma/src/asr/asr_preprocessing/src/switchboard/'

        self.__make()

    def __make(self):

        ####################
        # train (ldc97s62)
        ####################
        self.sph_train_paths = []
        for sph_path in glob.glob(os.path.join(self.train_data_path, '*/*/*/*.sph')):
            self.sph_train_paths.append(
                os.path.join(self.train_data_path, sph_path))

        trans_train_path = os.path.join(
            self.train_data_path, 'ldc97s62_new/transcriptions/swb_ms98_transcriptions')
        self.pronounce_dict_path = os.path.join(
            trans_train_path, 'sw-ms98-dict.text')

        self.word_train_paths = []
        self.trans_train_paths = []
        for text_path in glob.glob(os.path.join(trans_train_path, '*/*/*.text')):
            if text_path.split('.')[0][-4:] == 'word':
                self.word_train_paths.append(os.path.join(trans_train_path,
                                                          text_path))
            elif text_path.split('.')[0][-5:] == 'trans':
                self.trans_train_paths.append(os.path.join(trans_train_path,
                                                           text_path))

        ####################
        # train (fisher)
        ####################
        sph_train_fisher_path = os.path.join(
            self.train_data_fisher_path, 'fisher_english/audio/')
        trans_train_fisher_path = os.path.join(
            self.train_data_fisher_path, 'fisher_english/data/trans/')

        self.sph_train_fisher_paths = []
        self.trans_train_fisher_paths = []

        for sph_name in glob.glob(os.path.join(sph_train_fisher_path, '*/*.sph')):
            self.sph_train_fisher_paths.append(
                os.path.join(sph_train_fisher_path, sph_name))

        for text_name in glob.glob(os.path.join(trans_train_fisher_path, '*/*.txt')):
            self.trans_train_fisher_paths.append(
                os.path.join(trans_train_fisher_path, text_name))

        ########################################
        # test (eval2000, swbd + callhome)
        ########################################
        sph_test_path = os.path.join(
            self.test_data_path, 'ldc2002s09/english/')
        self.sph_test_paths = []
        self.sph_test_callhome_paths = []
        for sph_name in os.listdir(sph_test_path):
            if sph_name == 'hub5e_00.pem':
                # hub5e_00.pem file is a segmentation file
                pass
            elif sph_name[:2] == 'sw':
                self.sph_test_paths.append(
                    os.path.join(sph_test_path, sph_name))
            elif sph_name[:2] == 'en':
                self.sph_test_callhome_paths.append(
                    os.path.join(sph_test_path, sph_name))

        trans_test_path = os.path.join(
            self.test_data_path, 'ldc2002t43/reference/english/')
        self.trans_test_paths = []
        self.trans_test_callhome_paths = []
        for file_name in os.listdir(trans_test_path):
            # Switchboard evaluation transcript
            if file_name[:2] == 'sw':
                self.trans_test_paths.append(
                    os.path.join(trans_test_path, file_name))
            # CallHome evaluation transcript
            elif file_name[:2] == 'en':
                self.trans_test_callhome_paths.append(
                    os.path.join(trans_test_path, file_name))

    def sph_train(self):
        """Get paths to sph files of 300h training data (ldc97s62).
        Returns:
            paths to sph files
        """
        return sorted(self.sph_train_paths)

    def sph_train_fisher(self):
        """Get paths to sph files of 2000h training data (fisher).
        Returns:
            paths to sph files
        """
        return sorted(self.sph_train_fisher_paths)

    def sph_test(self):
        """Get paths to sph files of evaluation data (eval2000, swbd).
        Returns:
            paths to sph files
        """
        return sorted(self.sph_test_paths)

    def sph_test_callhome(self):
        """Get paths to sph files of evaluation data (eval2000, callhome).
        Returns:
            paths to sph files
        """
        return sorted(self.sph_test_callhome_paths)

    def label_train(self, label_type):
        """Get paths to transcription files of 300h training data (ldc97s62).
        Args:
            label_type: phone or character or word
        Returns:
            paths: paths to transcription files
        """
        if label_type in ['character', 'phone']:
            return sorted(self.trans_train_paths)
        elif label_type == 'word':
            return sorted(self.word_train_paths)

    def label_train_fisher(self):
        """Get paths to transcription files of 2000h training data (fisher).
        Returns:
            paths: paths to transcription files
        """
        return sorted(self.trans_train_fisher_paths)

    def label_test(self):
        """Get paths to transcription files of evaluation data (eval200, swbd).
        Returns:
            paths: paths to transcription files
        """
        return sorted(self.trans_test_paths)

    def label_test_callhome(self):
        """Get paths to transcription files of evaluation data (eval200, callhome).
        Returns:
            paths: paths to transcription files
        """
        return sorted(self.trans_test_callhome_paths)


if __name__ == '__main__':
    prep = Prepare()

    print('===== ldc97s62 ====')
    print(len(prep.sph_train()))
    print(len(prep.label_train(label_type='word')))
    print(len(prep.label_train(label_type='character')))

    print('==== fisher ====')
    print(len(prep.sph_train_fisher()))
    print(len(prep.label_train_fisher()))

    print('==== eval2000 (swbd) ====')
    print(len(prep.sph_test()))
    print(len(prep.label_test()))

    print('==== eval2000 (callhome) ====')
    print(len(prep.sph_test_callhome()))
    print(len(prep.label_test_callhome()))
