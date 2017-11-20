#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels for the End-to-End model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import re
from tqdm import tqdm

from utils.util import mkdir_join

# NOTE:
############################################################
# [character]
# 26 alphabets(a-z), space(_), apostorophe(')
# = 30 labels

# [character_capital_divide]
# - 100h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 19 special double-letters, apostorophe(')
# 74 labels

# - 460h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe(')
# = 79 labels

# - 960h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe(')
# = 79 labels

# [word]
# - 100h
# Original: 33798 labels + OOV
# - 460h
# Original: 65987 labels + OOV
# - 960h
# Original: 89114 labels + OOV
############################################################

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
APOSTROPHE = '\''
OOV = 'OOV'


def read_trans(label_paths, data_size, vocab_file_save_path, is_test=False,
               save_vocab_file=False, data_type=None):
    """Read transcript.
    Args:
        label_paths (list): list of paths to label files
        data_size (string): 100h or 460h or 960h
        vocab_file_save_path (string): path to vocabulary files
        is_test (bool, optional): if True, compute OOV rate
        save_vocab_file (bool, optional): if True, save vocabulary files
        data_type (string, optional): test_clean or test_other
    Returns:
        trans_dict (dict):
            key (string) => speaker-book-utt_index
            value (string) => transcript
    """
    print('===> Reading target labels...')
    speaker_dict = {}
    char_set, char_capital_set = set([]), set([])
    word_count_dict = {}
    vocab_set = set([])
    for label_path in tqdm(label_paths):
        speaker = label_path.split('/')[-3]
        if speaker not in speaker_dict.keys():
            speaker_dict[speaker] = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                utt_name = line[0]  # ex.) speaker-book-utt_index
                transcript = ' '.join(line[1:])
                word_list = line[1:]

                # Count words
                for word in word_list:
                    vocab_set.add(word)
                    if word not in word_count_dict.keys():
                        word_count_dict[word] = 0
                    word_count_dict[word] += 1

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

                speaker_dict[speaker][utt_name] = transcript

                # for debug
                # print(transcript)
                # print(transcript_capital_divide)
                # print('-----')

    # Make vocabulary files
    char_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'character_' + data_size + '.txt')
    char_capital_vocab_file_path = mkdir_join(
        vocab_file_save_path,
        'character_capital_divide_' + data_size + '.txt')
    word_freq1_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq1_' + data_size + '.txt')
    word_freq5_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq5_' + data_size + '.txt')
    word_freq10_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq10_' + data_size + '.txt')
    word_freq15_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq15_' + data_size + '.txt')

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
            char_list = sorted(list(char_capital_set)) + [APOSTROPHE]
            for char in char_list:
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

    # Compute OOV rate
    if is_test:
        with open(join(vocab_file_save_path, '../oov_rate_' + data_type + '_' + data_size + '.txt'), 'w') as f:

            # word-level (threshold == 1)
            oov_rate = compute_oov_rate(
                speaker_dict, word_freq1_vocab_file_path)
            f.write('Word (freq1):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 5)
            oov_rate = compute_oov_rate(
                speaker_dict, word_freq5_vocab_file_path)
            f.write('Word (freq5):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 10)
            oov_rate = compute_oov_rate(
                speaker_dict, word_freq10_vocab_file_path)
            f.write('Word (freq10):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 15)
            oov_rate = compute_oov_rate(
                speaker_dict, word_freq15_vocab_file_path)
            f.write('Word (freq15):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

    return speaker_dict


def compute_oov_rate(speaker_dict, vocab_file_path):

    with open(vocab_file_path, 'r') as f:
        vocab_set = set([])
        for line in f:
            word = line.strip()
            vocab_set.add(word)

    oov_count = 0
    word_num = 0
    for speaker_dict, utt_dict in speaker_dict.items():
        for utt_name, transcript in utt_dict.items():
            word_list = transcript.split(SPACE)
            word_num += len(word_list)

            for word in word_list:
                if word not in vocab_set:
                    oov_count += 1

    oov_rate = oov_count * 100 / word_num

    return oov_rate
