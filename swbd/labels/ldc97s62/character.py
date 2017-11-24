#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels for the End-to-End model (LDC97S62 corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm
from collections import OrderedDict

from swbd.labels.ldc97s62.fix_trans import fix_transcript
from utils.labels.character import Char2idx
from utils.labels.word import Word2idx
from utils.util import mkdir_join

# NOTE:
############################################################
# [character]
# 26 alphabets(a-z), number(0-9), space(_), apostorophe('), hyphen(-)
# laughter(L), noise(N), vocalized-noise(V)
# = 42 labels

# [character_capital_divide]
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 22 special double-letters, apostorophe('), hyphen(-),
# laughter(L), noise(N), vocalized-noise(V)
# = 92 labels

# [word]
# Original: ? labels + OOV
############################################################

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
HYPHEN = '-'
APOSTROPHE = '\''
LAUGHTER = 'LA'
NOISE = 'NZ'
VOCALIZED_NOISE = 'VN'
OOV = 'OOV'


def read_trans(label_paths, run_root_path, vocab_file_save_path,
               save_vocab_file=False,  speaker_dict_fisher=None,
               char_set=None, char_capital_set=None, word_count_dict=None):
    """Read transcripts (*_trans.txt) & save files (.npy).
    Args:
        label_paths (list): list of paths to label files
        run_root_path (string):
        vocab_file_save_path (string): path to vocabulary files
        save_vocab_file (bool, optional): if True, save vocabulary files
        speaker_dict_fisher (dict):
        char_set (set):
        char_capital_set (set):
        word_count_dict (dict):
    Returns:
        speaker_dict: dictionary of speakers
            key (string) => speaker
            value (dict) => dictionary of utterance infomation of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, char_indices, char_indices_capital,
                                word_freq1_indices, word_freq5_indices,
                                word_freq10_indices, word_freq15_indices]
    """
    print('=====> Processing target labels...')
    merge_with_fisher = True if speaker_dict_fisher is not None else False

    if merge_with_fisher:
        speaker_dict = speaker_dict_fisher
        vocab_set = set([])
        for word in word_count_dict.keys():
            vocab_set.add(word)
    else:
        speaker_dict = OrderedDict()
        char_set, char_capital_set = set([]), set([])
        word_count_dict = {}
        vocab_set = set([])

    for label_path in tqdm(label_paths):
        utterance_dict = OrderedDict()
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                speaker = line[0].split('-')[0]
                # Fix speaker name
                speaker = speaker.replace('sw0', 'sw').replace(
                    'a', '-A').replace('b', '-B')
                utt_index = line[0].split('-')[-1]
                start_frame = int(float(line[1]) * 100 + 0.05)
                end_frame = int(float(line[2]) * 100 + 0.05)
                transcript = ' '.join(line[3:])

                # Clean transcript
                transcript = fix_transcript(transcript)

                # Skip silence
                if transcript in ['', ' ']:
                    continue

                # Convert space to "_"
                transcript = re.sub(r'\s', SPACE, transcript)

                # Skip laughter, noise, vocalized-noise only utterance
                if transcript.replace(NOISE, '').replace(LAUGHTER, '').replace(VOCALIZED_NOISE, '').replace(SPACE, '') != '':

                    # Remove the first and last space
                    if transcript[0] == SPACE:
                        transcript = transcript[1:]
                    if transcript[-1] == SPACE:
                        transcript = transcript[:-1]

                    # Count words
                    for word in transcript.split(SPACE):
                        vocab_set.add(word)
                        if word not in word_count_dict.keys():
                            word_count_dict[word] = 0
                        word_count_dict[word] += 1

                    # Capital-divided
                    transcript_capital = ''
                    for word in transcript.split(SPACE):
                        if len(word) == 1:
                            char_capital_set.add(word)
                            transcript_capital += word
                        else:
                            # Replace the first character with the capital
                            # letter
                            word = word[0].upper() + word[1:]

                            # Check double-letters
                            for i in range(0, len(word) - 1, 1):
                                if word[i:i + 2] in DOUBLE_LETTERS:
                                    char_capital_set.add(word[i:i + 2])
                                else:
                                    char_capital_set.add(word[i])
                            transcript_capital += word

                    for c in list(transcript):
                        char_set.add(c)

                    utterance_dict[utt_index.zfill(4)] = [
                        start_frame, end_frame, transcript]

                    # for debug
                    # print(transcript_original)
                    # print(transcript)
                    # print(transcript_capital)

            speaker_dict[speaker] = utterance_dict

    # Make vocabulary files
    data_size = '2000h' if merge_with_fisher else '300h'
    char_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'character_' + data_size + '.txt')
    char_capital_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'character_capital_divide_' + data_size + '.txt')
    word_freq1_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq1_' + data_size + '.txt')
    word_freq5_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq5_' + data_size + '.txt')
    word_freq10_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq10_' + data_size + '.txt')
    word_freq15_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq15_' + data_size + '.txt')

    # Reserve some indices
    for mark in [SPACE, HYPHEN, APOSTROPHE, LAUGHTER, NOISE, VOCALIZED_NOISE]:
        for c in list(mark):
            char_set.discard(c)
    for mark in [SPACE, HYPHEN, APOSTROPHE]:
        for c in list(mark):
            char_capital_set.discard(c)

    # for debug
    # print(sorted(list(char_set)))
    # print(sorted(list(char_capital_set)))

    if save_vocab_file:
        # character-level
        with open(char_vocab_file_path, 'w') as f:
            char_list = sorted(list(char_set)) + \
                [SPACE, APOSTROPHE, HYPHEN, LAUGHTER, NOISE, VOCALIZED_NOISE]
            for char in char_list:
                f.write('%s\n' % char)

        # character-level (capital-divided)
        with open(char_capital_vocab_file_path, 'w') as f:
            char_capital_list = sorted(list(char_capital_set)) + \
                [APOSTROPHE, HYPHEN, LAUGHTER, NOISE, VOCALIZED_NOISE]
            for char in char_capital_list:
                f.write('%s\n' % char)

        # word-level (threshold == 1)
        with open(word_freq1_vocab_file_path, 'w') as f:
            vocab_list = sorted(list(vocab_set)) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 5)
        with open(word_freq5_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 5]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 10)
        with open(word_freq10_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 10]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 15)
        with open(word_freq15_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 15]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

    # Tokenize
    print('=====> Tokenize...')
    char2idx = Char2idx(char_vocab_file_path, double_letter=True)
    char2idx_capital = Char2idx(
        char_capital_vocab_file_path, capital_divide=True)
    word2idx_freq1 = Word2idx(word_freq1_vocab_file_path)
    word2idx_freq5 = Word2idx(word_freq5_vocab_file_path)
    word2idx_freq10 = Word2idx(word_freq10_vocab_file_path)
    word2idx_freq15 = Word2idx(word_freq15_vocab_file_path)
    for speaker, utt_dict in tqdm(speaker_dict.items()):
        for utt_index, [start_frame, end_frame, transcript] in utt_dict.items():
            char_indices = char2idx(transcript)
            char_indices_capital = char2idx_capital(transcript)
            word_freq1_indices = word2idx_freq1(transcript)
            word_freq5_indices = word2idx_freq5(transcript)
            word_freq10_indices = word2idx_freq10(transcript)
            word_freq15_indices = word2idx_freq15(transcript)

            char_indices = ' '.join(list(map(str, char_indices.tolist())))
            char_indices_capital = ' '.join(
                list(map(str, char_indices_capital.tolist())))
            word_freq1_indices = ' '.join(
                list(map(str, word_freq1_indices.tolist())))
            word_freq5_indices = ' '.join(
                list(map(str, word_freq5_indices.tolist())))
            word_freq10_indices = ' '.join(
                list(map(str, word_freq10_indices.tolist())))
            word_freq15_indices = ' '.join(
                list(map(str, word_freq15_indices.tolist())))

            utt_dict[utt_index] = [start_frame, end_frame,
                                   char_indices, char_indices_capital,
                                   word_freq1_indices, word_freq5_indices,
                                   word_freq10_indices, word_freq15_indices]
        speaker_dict[speaker] = utt_dict

    return speaker_dict
