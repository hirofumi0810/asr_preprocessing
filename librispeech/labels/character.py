#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level target labels for the End-to-End model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import re
import numpy as np
from tqdm import tqdm

from utils.labels.character import Char2idx
from utils.util import mkdir_join

# NOTE:
############################################################
# CTC model

# [character]
# 26 alphabets(a-z), space(_), apostorophe(')
# = 26 + 2 = 28 labels

# [character_capital_divide]
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe(')
# = 26 * 2 + 24 + 1 = 77 labels
############################################################

############################################################
# Attention-based model

# [character]
# 26 alphabets(a-z), space(_), apostorophe('), <SOS>, <EOS>
# = 26 + 2 + 2 = 30 labels

# [character_capital_divide]
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe('), <SOS>, <EOS>
# = 26 * 2 + 24 + 1 + 2 = 79 labels
############################################################


def read_char(label_paths, run_root_path, model, save_map_file=False,
              save_path=None, divide_by_capital=False,
              stdout_transcript=False):
    """Read text transcript.
    Args:
        label_paths (list): list of paths to label files
        run_root_path (string): absolute path of make.sh
        model (string): ctc or attention
        save_map_file (bool, optional): if True, save the mapping file
        save_path (string, optional): path to save labels. If None, don't save
            labels
        divide_by_capital (bool, optional): if True, each word will be diveded
            by capital letters rather than spaces. In addition, repeated
            letters will be grouped by a special double-letter unit.
                ex.) hello => h e ll o
            This implementation is based on
                https://arxiv.org/abs/1609.05935.
                    Zweig, Geoffrey, et al.
                    "Advances in all-neural speech recognition."
                    in Proceedings of ICASSP, 2017.
        stdout_transcript (bool, optional): if True, print processed
            transcripts to standard output
    """
    if model not in ['ctc', 'attention']:
        raise ValueError('model must be ctc or attention.')

    print('===> Reading target labels...')
    speaker_dict = {}
    char_set = set([])
    for label_path in tqdm(label_paths):
        speaker = label_path.split('/')[-3]
        if speaker not in speaker_dict.keys():
            speaker_dict[speaker] = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                utt_name = line[0]  # ex.) speaker-book-utt_index
                transcript = ' '.join(line[1:])

                if divide_by_capital:
                    transcript_capital = ''
                    for word in transcript.split(' '):
                        # Replace space with a capital letter
                        word = word[0].upper() + word[1:]

                        # Check double-letters
                        for i in range(0, len(word) - 1, 1):
                            if word[i] == word[i + 1]:
                                char_set.add(word[i] * 2)

                        transcript_capital += word
                    transcript = transcript_capital
                else:
                    # Convert space to "_"
                    transcript = re.sub(r'\s', '_', transcript)

                for c in list(transcript):
                    char_set.add(c)

                if model == 'attention':
                    transcript = '<' + transcript + '>'

                speaker_dict[speaker][utt_name] = transcript

                if stdout_transcript:
                    print(transcript)

    # Make mapping file (from character to index)
    if divide_by_capital:
        map_file_path = mkdir_join(
            run_root_path, 'labels', 'mapping_files', model,
            'character_capital_divide.txt')
    else:
        map_file_path = mkdir_join(
            run_root_path, 'labels', 'mapping_files', model,
            'character.txt')
    char_set.discard('_')
    char_set.discard('\'')
    if model == 'attention':
        char_set.discard('<')
        char_set.discard('>')

    # for debug
    # print(sorted(list(char_set)))

    if save_map_file:
        with open(map_file_path, 'w') as f:
            if model == 'ctc':
                if divide_by_capital:
                    char_list = sorted(list(char_set)) + ['\'']
                else:
                    char_list = ['_'] + sorted(list(char_set)) + ['\'']
            elif model == 'attention':
                if divide_by_capital:
                    char_list = ['<', '>'] + sorted(list(char_set)) + ['\'']
                else:
                    char_list = ['_', '<', '>'] + \
                        sorted(list(char_set)) + ['\'']
            for i, char in enumerate(char_list):
                f.write('%s  %s\n' % (char, str(i)))

    if save_path is not None:
        char2idx = Char2idx(map_file_path=map_file_path)

        # Save target labels
        print('===> Saving target labels...')
        for speaker, utterance_dict in tqdm(speaker_dict.items()):
            for utt_name, transcript in utterance_dict.items():

                # Convert from character to index
                char_index_list = char2idx(
                    transcript, double_letter=divide_by_capital)

                # Save as npy file
                np.save(mkdir_join(save_path, speaker, utt_name + '.npy'),
                        char_index_list)
