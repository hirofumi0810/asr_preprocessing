#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level target labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import re
from tqdm import tqdm

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


def read_char(label_paths, vocab_file_save_path, save_vocab_file=False):
    """Read text transcript.
    Args:
        label_paths (list): list of paths to label files
        vocab_file_save_path (string): path to vocabulary files
        save_vocab_file (string): if True, save vocabulary files
    Returns:
        trans_dict (dict):
            key (string) => utterance name
            value (string) => transcript
    """
    print('=====> Reading target labels...')
    trans_dict = {}
    char_set, char_capital_set = set([]), set([])
    for label_path in tqdm(label_paths):
        with open(label_path, 'r') as f:
            line = f.readlines()[-1]
            speaker = label_path.split('/')[-2]
            utt_index = basename(label_path).split('.')[0]
            utt_name = speaker + '_' + utt_index

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

            # Capital-divided
            for word in transcript.split(' '):
                if len(word) == 1:
                    char_capital_set.add(word.upper())
                else:
                    # Replace the first character with the capital letter
                    word = word[0].upper() + word[1:]
                    char_capital_set.add(word[0].upper())

                    # Check double-letters
                    skip_flag = False
                    for i in range(1, len(word) - 1, 1):
                        if skip_flag:
                            skip_flag = False
                            continue

                        if not skip_flag and word[i:i + 2] in DOUBLE_LETTERS:
                            char_capital_set.add(word[i:i + 2])
                            skip_flag = True
                        else:
                            char_capital_set.add(word[i])

                    # Final character
                    if not skip_flag:
                        char_capital_set.add(word[-1])

            # Convert space to "_"
            transcript = re.sub(r'\s', SPACE, transcript)

            for c in list(transcript):
                char_set.add(c)

            trans_dict[utt_name] = transcript

            # for debug
            # print(transcript)
            # print(trans_char_capital_divide)

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

    return trans_dict
