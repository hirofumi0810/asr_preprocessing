#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make word-level target labels for the End-to-End model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from os.path import join
import numpy as np
from tqdm import tqdm

from utils.labels.phone import phone2idx
from utils.util import mkdir_join

# NOTE:
############################################################
# CTC model

############################################################

############################################################
# Attention-based model

############################################################


def read_phone(label_paths, data_type, run_root_path, model,
               lexicon_path, save_map_file=False, save_path=None,
               stdout_transcript=False):
    """Read text transcript.
    Args:
        label_paths (list): list of paths to label files
        data_type (string): train_clean100 or train_clean360 or train_other500
            or train_all or dev_clean or dev_other or test_clean or test_clean
        run_root_path (string): absolute path of make.sh
        model (string): ctc or attention
        lexicon_path: (string): path to the lexicon
        save_map_file (bool ,optional): if True, save the mapping file
        save_path (string, optional): path to save labels. If None, don't save
            labels
        stdout_transcript (bool, optional): if True, print processed
            transcripts to standard output
    """
    if model not in ['ctc', 'attention']:
        raise ValueError('model must be ctc or attention.')

    print('Reading lexicon...')
    word2phone = {}
    phone_set = set([])
    with open(join(lexicon_path), 'r') as f:
        for line in f:
            line = line.strip().lower()
            # Remove tab and consecutive spaces
            line = re.sub(r'[\t]+', ' ', line)
            line = re.sub(r'[\s]+', ' ', line)
            if 'cuthbert' in line:
                print(line)
            if 'unsup' in line:
                print(line)
            if 'hopet' in line:
                print(line)
            line = line.split(' ')
            while '' in line:
                line.remove('')
            word = line[0]
            phone_seq = ' '.join(line[1:])
            word2phone[word] = phone_seq
            for phone in phone_seq.split(' '):
                phone_set.add(phone)

    print('===> Reading target labels...')
    speaker_dict = {}
    for label_path in tqdm(label_paths):
        speaker = label_path.split('/')[-3]
        if speaker not in speaker_dict.keys():
            speaker_dict[speaker] = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                # print
                utt_name = line[0]  # ex.) speaker-book-utt_index
                word_list = line[1:]

                # Convert from word to phone
                try:
                    phone_list = [word2phone[word] for word in word_list]
                except:
                    phone_list = []
                    for word in word_list:
                        if word == 'cuthberts':
                            phone_list.append(word2phone['cuthbert\'s'])
                        elif word == 'unsupposable':
                            phone_list.append(word2phone['unsupportable'])
                        elif word == 'hopeton':
                            phone_list.append(word2phone['hopetoun'])
                        elif word in word2phone.keys():
                            phone_list.append(word2phone[word])
                        else:
                            print(word)

                if model == 'attention':
                    phone_list = ['<'] + phone_list + ['>']

                speaker_dict[speaker][utt_name] = phone_list

                # if stdout_transcript:
                #     print(' '.join(phone_list))

    mapping_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', model, 'phone.txt')

    # Make mapping file (from phone to index)
    if save_map_file:
        all_phone_list = sorted(list(phone_set))
        if model == 'attention':
            all_phone_list = ['<', '>'] + all_phone_list
        with open(mapping_file_path, 'w') as f:
            for i, phone in enumerate(all_phone_list):
                f.write('%s  %s\n' % (phone, str(i)))

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for speaker, utterance_dict in tqdm(speaker_dict.items()):
            for utt_name, phone_list in utterance_dict.items():

                # Convert from word to index
                phone_index_list = phone2idx(phone_list, mapping_file_path)

                # Save as npy file
                np.save(mkdir_join(save_path, speaker, utt_name + '.npy'),
                        phone_index_list)
