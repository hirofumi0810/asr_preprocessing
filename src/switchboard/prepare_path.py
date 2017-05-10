#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare for making dataset."""

import os
import glob


class Prepare(object):
    """Prepare for making dataset."""

    def __init__(self):

        # path to save dataset (set yourself)
        self.data_root_path = '/n/sd8/inaguma/corpus/switchboard/'
        self.run_root_path = '/n/sd8/inaguma/src/asr/asr_preprocessing/asr_preprocessing/corpus/switchboard/'

        # path to switchboard path (set yourself)
        self.train_data_path = '/n/sd8/inaguma/corpus/switchboard/LDC97S62/'
        self.train_data_fisher_path = '/n/sd8/inaguma/corpus/switchboard/fisher/'
        self.test_data_path = '/n/sd8/inaguma/corpus/switchboard/eval2000/'  # swbd + callhome

        self.__make()

    def __make(self):

        ####################
        # train (LDC97S62)
        ####################
        sph_train_dir_list = []
        sph_train_dir_list.append(os.path.join(self.train_data_path, 'swb1_d1/data/'))
        sph_train_dir_list.append(os.path.join(self.train_data_path, 'swb1_d2/data/'))
        sph_train_dir_list.append(os.path.join(self.train_data_path, 'swb1_d3/data/'))
        sph_train_dir_list.append(os.path.join(self.train_data_path, 'swb1_d4/data/'))
        transcript_train_path = os.path.join(
            self.train_data_path, 'transcriptions/swb_ms98_transcriptions')
        self.pronounce_dict_path = os.path.join(transcript_train_path, 'sw-ms98-dict.text')

        self.sph_train_paths = []
        for sph_train_dir in sph_train_dir_list:
            for sph_name in os.listdir(sph_train_dir):
                self.sph_train_paths.append(os.path.join(sph_train_dir, sph_name))

        self.word_train_paths = []
        self.transcript_train_paths = []
        for dir_name in os.listdir(transcript_train_path):
            if dir_name.split('.')[-1] != 'text':
                for session_name in os.listdir(os.path.join(transcript_train_path, dir_name)):
                    for txt_name in os.listdir(os.path.join(transcript_train_path, dir_name, session_name)):
                        if len(os.listdir(os.path.join(transcript_train_path, dir_name, session_name))) != 4:
                            print(os.path.join(transcript_train_path, dir_name, session_name))
                        if txt_name.split('.')[0][-4:] == 'word':
                            self.word_train_paths.append(os.path.join(transcript_train_path,
                                                                      dir_name,
                                                                      session_name,
                                                                      txt_name))
                        elif txt_name.split('.')[0][-5:] == 'trans':
                            self.transcript_train_paths.append(os.path.join(transcript_train_path,
                                                                            dir_name,
                                                                            session_name,
                                                                            txt_name))

        ####################
        # train (Fisher)
        ####################
        sph_train_fisher_path = os.path.join(self.train_data_fisher_path, 'fisher_english/audio/')
        transcript_train_fisher_path = os.path.join(
            self.train_data_fisher_path, 'fisher_english/data/trans/')

        self.sph_train_fisher_paths = []
        self.transcript_train_fisher_paths = []

        for sph_name in glob.glob(os.path.join(sph_train_fisher_path, '*/*.sph')):
            self.sph_train_fisher_paths.append(os.path.join(sph_train_fisher_path, sph_name))

        for text_name in glob.glob(os.path.join(transcript_train_fisher_path, '*/*.txt')):
            self.transcript_train_fisher_paths.append(
                os.path.join(transcript_train_fisher_path, text_name))

        ########################################
        # test (eval2000, swbd + callhome)
        ########################################
        sph_test_path = os.path.join(self.test_data_path, 'LDC2002S09/english/')
        self.sph_test_paths = []
        self.sph_test_callhome_paths = []
        for sph_name in os.listdir(sph_test_path):
            if sph_name == 'hub5e_00.pem':
                # hub5e_00.pem file is a segmentation file
                pass
            elif sph_name[:2] == 'sw':
                self.sph_test_paths.append(os.path.join(sph_test_path, sph_name))
            elif sph_name[:2] == 'en':
                self.sph_test_callhome_paths.append(os.path.join(sph_test_path, sph_name))

        transcript_test_path = os.path.join(self.test_data_path, 'LDC2002T43/reference/english/')
        self.transcript_test_paths = []
        self.transcript_test_callhome_paths = []
        for file_name in os.listdir(transcript_test_path):
            # Switchboard evaluation transcript
            if file_name[:2] == 'sw':
                self.transcript_test_paths.append(os.path.join(transcript_test_path, file_name))
            # CallHome evaluation transcript
            elif file_name[:2] == 'en':
                self.transcript_test_callhome_paths.append(
                    os.path.join(transcript_test_path, file_name))

    def sph_train(self):
        """Get paths to sph files of 300h training data (LDC97S62).
        Returns:
            paths to sph files
        """
        return sorted(self.sph_train_paths)

    def sph_train_fisher(self):
        """Get paths to sph files of 2000h training data (Fisher).
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
        """Get paths to transcription files of 300h training data (LDC97S62).
        Args:
            label_type: phone or character or word
        Returns:
            paths: paths to transcription files
        """
        if label_type in ['character', 'phone']:
            return sorted(self.transcript_train_paths)
        elif label_type == 'word':
            return sorted(self.word_train_paths)

    def label_train_fisher(self):
        """Get paths to transcription files of 2000h training data (Fisher).
        Returns:
            paths: paths to transcription files
        """
        return sorted(self.transcript_train_fisher_paths)

    def label_test(self):
        """Get paths to transcription files of evaluation data (eval200, swbd).
        Returns:
            paths: paths to transcription files
        """
        return sorted(self.transcript_test_paths)

    def label_test_callhome(self):
        """Get paths to transcription files of evaluation data (eval200, callhome).
        Returns:
            paths: paths to transcription files
        """
        return sorted(self.transcript_test_callhome_paths)


if __name__ == '__main__':
    prep = Prepare()

    print('===== LDC97S62 ====')
    print(len(prep.sph_train()))
    print(len(prep.label_train(label_type='word')))
    print(len(prep.label_train(label_type='character')))

    print('==== Fisher ====')
    print(len(prep.sph_train_fisher()))
    print(len(prep.label_train_fisher()))

    print('==== eval2000 (swbd) ====')
    print(len(prep.sph_test()))
    print(len(prep.label_test()))

    print('==== eval2000 (callhome) ====')
    print(len(prep.sph_test_callhome()))
    print(len(prep.label_test_callhome()))
