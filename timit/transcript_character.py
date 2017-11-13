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
# [character]
# 26 alphabets(a-z)
# space(_), apostorophe(')
# = 30 labels

# [character_capital_divide]
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 19 special double-letters, apostorophe(')
# = 74 labels
############################################################

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
APOSTROPHE = '\''


def read_char(label_paths, vocab_file_save_path, is_test=False,
              save_vocab_file=False, save_path=None):
    """Read text transcript.
    Args:
        label_paths (list): list of paths to label files
        vocab_file_save_path (string): path to vocabulary files
        is_test (bool, optional): Set True when proccessing the test set
        save_vocab_file (string): if True, save vocabulary files
        save_path (string, optional): path to save labels.
            If None, don't save labels.
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

    # Make vocabulary files
    char_vocab_file_path = mkdir_join(vocab_file_save_path, 'character.txt')
    char_capital_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'character_capital_divide.txt')

    # Reserve some indices
    char_set.discard(SPACE)
    char_set.discard(APOSTROPHE)
    char_capital_set.discard(APOSTROPHE)

    # for debug
    # print(sorted(list(char_set)))
    # print(sorted(list(char_capital_set)))

    if save_vocab_file:
        # character-level
        with open(char_vocab_file_path, 'w') as f:
            char_list = sorted(list(char_set)) + [SPACE, APOSTROPHE]
            for char in char_list:
                f.write('%s\n' % char)

        # character-level (capital-divided)
        with open(char_capital_vocab_file_path, 'w') as f:
            char_capital_list = sorted(list(char_capital_set)) + [APOSTROPHE]
            for char in char_capital_list:
                f.write('%s\n' % char)

    if not is_test:
        char2idx = Char2idx(vocab_file_path=char_vocab_file_path)
        char2idx_capital = Char2idx(vocab_file_path=char_capital_vocab_file_path,
                                    double_letter=True)

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for label_path, [transcript, transcript_capital] in tqdm(text_dict.items()):
            speaker = label_path.split('/')[-2]
            utt_index = basename(label_path).split('.')[0]
            save_file_name = speaker + '_' + utt_index + '.npy'

            if is_test:
                # Save target labels as string
                np.save(mkdir_join(save_path, 'character',
                                   save_file_name), transcript)
                np.save(mkdir_join(save_path, 'character_capital_divide',
                                   save_file_name), transcript)
            else:
                # Save target labels as index
                np.save(mkdir_join(save_path, 'character', save_file_name),
                        char2idx(transcript))
                np.save(mkdir_join(save_path, 'character_capital_divide',
                                   save_file_name),
                        char2idx_capital(transcript_capital))
