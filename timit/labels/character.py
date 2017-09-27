#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level target labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import re
import numpy as np
from tqdm import tqdm
import pickle

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


def read_text(label_paths, run_root_path, model, save_map_file=False,
              save_path=None, divide_by_capital=False, stdout_transcript=False):
    """Read text transcript.
    Args:
        label_paths (list): list of paths to label files
        run_root_path (string): absolute path of make.sh
        model (string): ctc or attention
        save_map_file (string): if True, save the mapping file
        save_path (string, optional): path to save labels. If None, don't save labels
        divide_by_capital (bool, optional): if True, each word will be diveded
            by capital letters rather than spaces. In addition, repeated letters
            will be grouped by a special double-letter unit.
                ex.) hello => h e ll o
            This implementation is based on
                https://arxiv.org/abs/1609.05935.
                    Zweig, Geoffrey, et al.
                    "Advances in all-neural speech recognition."
                    in Proceedings of ICASSP, 2017.
        stdout_transcript (bool, optional): if True, print transcripts to standard output
    """
    if model not in ['ctc', 'attention']:
        raise TypeError('model must be ctc or attention.')

    print('===> Reading target labels...')
    text_dict = {}
    char_set = set([])
    for label_path in tqdm(label_paths):
        with open(label_path, 'r') as f:
            line = f.readlines()[-1]

            # Remove 「"」, 「:」, 「;」, 「！」, 「?」, 「,」, 「.」, 「-」
            # Convert to lowercase
            line = re.sub(r'[\":;!?,.-]+', '', line.strip().lower())

            if divide_by_capital:
                transcript = ''
                for word in line.split(' ')[2:]:
                    if len(word) == 0:
                        continue

                    # Replace space with a capital letter
                    word = word[0].upper() + word[1:]

                    # Check double-letters
                    for i in range(0, len(word) - 1, 1):
                        if word[i] == word[i + 1]:
                            char_set.add(word[i] * 2)

                    transcript += word

            else:
                # Convert space to "_"
                transcript = '_'.join(line.split(' ')[2:])

            if model == 'attention':
                transcript = '<' + transcript + '>'

        for c in list(transcript):
            char_set.add(c)

        text_dict[label_path] = transcript

        # for debug
        if stdout_transcript:
            print(transcript)

    # Make mapping file (from character to number)
    if divide_by_capital:
        map_file_path = mkdir_join(
            run_root_path, 'labels', 'mapping_files', model, 'character_capital.txt')
    else:
        map_file_path = mkdir_join(
            run_root_path, 'labels', 'mapping_files', model, 'character.txt')
    char_set.discard('_')
    char_set.discard('\'')
    if model == 'attention':
        char_set.discard('<')
        char_set.discard('>')

    if save_map_file:
        with open(map_file_path, 'w') as f:
            if model == 'attention':
                char_list = ['<', '>']
            elif model == 'ctc':
                char_list = []

            if divide_by_capital:
                char_list += sorted(list(char_set)) + ['\'']
            else:
                char_list += ['_'] + sorted(list(char_set)) + ['\'']

            for i, char in enumerate(char_list):
                f.write('%s  %s\n' % (char, str(i)))

    if save_path is not None:
        char2idx = Char2idx(map_file_path=map_file_path)
        label_num_dict = {}

        # Save target labels
        print('===> Saving target labels...')
        for label_path, transcript in tqdm(text_dict.items()):
            speaker = label_path.split('/')[-2]
            utt_index = basename(label_path).split('.')[0]

            # Convert from character to index
            index_list = char2idx(transcript, double_letter=divide_by_capital)

            # Save as npy file
            np.save(mkdir_join(save_path, speaker + '_' +
                               utt_index + '.npy'), index_list)

            # Count label number
            label_num_dict[speaker + '_' + utt_index] = len(index_list)

        # Save the label number dictionary
        with open(join(save_path, 'label_num.pickle'), 'wb') as f:
            pickle.dump(label_num_dict, f)
