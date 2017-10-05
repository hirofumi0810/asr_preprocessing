#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level target labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import re
import numpy as np
from tqdm import tqdm

from utils.labels.character import Char2idx
from utils.util import mkdir_join

# NOTE:
############################################################
# CTC model

# - character
# 26 alphabets(a-z),
# space(_), apostorophe('),
# = 26 + 2 = 28 labels

# - character_capital_divide
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 19 special double-letters, apostorophe(')
# = 26 * 2 + 19 + 1 = 72 labels
############################################################

############################################################
# Attention-based model

# - character
# 26 alphabets(a-z), <SOS>, <EOS>
# space(_), apostorophe(')
# = 26 + 2 + 2 = 30 labels

# - character_capital_divide
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z)  <SOS>, <EOS>
# 19 special double-letters, apostorophe(')
# = 26 * 2 + 2 + 19 + 1 = 74 labels
############################################################

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
APOSTROPHE = '\''
SOS = '<'
EOS = '>'


def read_char(label_paths, run_root_path, save_map_file=False,
              ctc_char_save_path=None, att_char_save_path=None,
              ctc_char_capital_save_path=None, att_char_capital_save_path=None):
    """Read text transcript.
    Args:
        label_paths (list): list of paths to label files
        run_root_path (string): absolute path of make.sh
        save_map_file (string): if True, save the mapping file
        ctc_char_save_path (string, optional): path to save character-level
            labels for the CTC models. If None, don't save labels.
        ctc_char_capital_save_path (string, optional): path to save
            capital-divided character-level labels for the CTC models.
            If None, don't save labels.
        att_char_save_path (string, optional): path to save character-level
            labels for the Attention-based models. If None, don't save labels.
        att_char_capital_save_path (string, optional): path to save
            capital-divided character-level labels for the Attention-based
            models. If None, don't save labels.
    """
    print('===> Reading target labels...')
    text_dict = {}
    char_set, char_capital_set = set([]), set([])
    for label_path in tqdm(label_paths):
        with open(label_path, 'r') as f:
            line = f.readlines()[-1]

            # Remove 「"」, 「:」, 「;」, 「！」, 「?」, 「,」, 「.」, 「-」
            # Convert to lowercase
            line = re.sub(r'[\":;!?,.-]+', '', line.strip().lower())

            transcript = ' '.join(line.split(' ')[2:])

            # Remove double spaces
            while '  ' in transcript:
                transcript = re.sub(r'  ', ' ', transcript)

            # Remove first and last space
            if transcript[0] == ' ':
                transcript = transcript[1:]
            if transcript[-1] == ' ':
                transcript = transcript[:-1]

            # capital-divided version
            transcript_capital = ''
            for word in transcript.split(' '):
                if len(word) == 1:
                    char_capital_set.add(word)
                    transcript_capital += word
                else:
                    # Replace the first character with the capital letter
                    word = word[0].upper() + word[1:]

                    # Check double-letters
                    for i in range(0, len(word) - 1, 1):
                        if word[i:i + 2] in DOUBLE_LETTERS:
                            char_capital_set.add(word[i] * 2)
                        else:
                            char_capital_set.add(word[i])
                    transcript_capital += word

            # Convert space to "_"
            transcript = re.sub(r'\s', SPACE, transcript)

            for c in list(transcript):
                char_set.add(c)

        text_dict[label_path] = [transcript, transcript_capital]

        # for debug
        # print(transcript)
        # print(transcript_capital)

    # Make mapping file (from character to index)
    char_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', 'character.txt')
    char_capital_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', 'character_capital_divide.txt')

    # Reserve some indices
    char_set.discard(SPACE)
    char_set.discard(APOSTROPHE)
    char_capital_set.discard(APOSTROPHE)

    if save_map_file:
        with open(char_map_file_path, 'w') as f:
            char_list = [SPACE] + \
                sorted(list(char_set)) + [APOSTROPHE, SOS, EOS]

            for i, char in enumerate(char_list):
                f.write('%s  %s\n' % (char, str(i)))

        with open(char_capital_map_file_path, 'w') as f:
            char_capital_list = sorted(
                list(char_capital_set)) + [APOSTROPHE, SOS, EOS]

            for i, char in enumerate(char_capital_list):
                f.write('%s  %s\n' % (char, str(i)))

    char2idx = Char2idx(map_file_path=char_map_file_path)
    char2idx_capital = Char2idx(map_file_path=char_capital_map_file_path)

    # Save target labels
    print('===> Saving target labels...')
    for label_path, [transcript, transcript_capital] in tqdm(text_dict.items()):
        speaker = label_path.split('/')[-2]
        utt_index = basename(label_path).split('.')[0]
        save_file_name = speaker + '_' + utt_index + '.npy'

        if ctc_char_save_path is not None:
            # Convert from character to index
            index_list = char2idx(transcript, double_letter=False)

            # Save as npy file
            np.save(mkdir_join(ctc_char_save_path, save_file_name), index_list)

        if att_char_save_path is not None:
            # Convert from character to index
            index_list = char2idx(
                SOS + transcript + EOS, double_letter=False)

            # Save as npy file
            np.save(mkdir_join(att_char_save_path, save_file_name), index_list)

        if ctc_char_capital_save_path is not None:
            # Convert from character to index
            index_list = char2idx_capital(
                transcript_capital, double_letter=True)

            # Save as npy file
            np.save(mkdir_join(ctc_char_capital_save_path,
                               save_file_name), index_list)

        if att_char_capital_save_path is not None:
            # Convert from character to index
            index_list = char2idx_capital(
                SOS + transcript_capital + EOS, double_letter=True)

            # Save as npy file
            np.save(mkdir_join(att_char_capital_save_path,
                               save_file_name), index_list)
