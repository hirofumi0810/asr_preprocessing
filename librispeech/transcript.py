#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels for the End-to-End model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import isfile
import re
import numpy as np
from tqdm import tqdm

from utils.labels.word import Word2idx
from utils.labels.character import Char2idx
from utils.util import mkdir_join

# NOTE:
############################################################
# [phone]
#

# [character]
# 26 alphabets(a-z), space(_), apostorophe('), <SOS>, <EOS>
# = 30 labels

# [character_capital_divide]
# - train100h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 19 special double-letters, apostorophe('), <SOS>, <EOS>
# 74 labels

# - train460h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe('), <SOS>, <EOS>
# = 79 labels

# - train960h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe('), <SOS>, <EOS>
# = 79 labels

# [word, threshold == 10]
# - train100h
# Original: 33798 labels (+2)
# Restricted: 7213 labels (+2)
# OOV rate: 6.85%

# - train460h
# Original:  labels (+2)
# Restricted:  labels (+2)
# OOV rate: %

# - train960h
# Original: 89114 labels (+2)
# Restricted: 26642 labels (+2)
# OOV rate: 1.67%
############################################################

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
APOSTROPHE = '\''
SOS = '<'
EOS = '>'
OOV = 'OOV'


def read_trans(label_paths, train_data_size, map_file_save_path,
               is_training=False, is_test=False, frequency_threshold=5,
               save_map_file=False, save_path=None):
    """Read transcript.
    Args:
        label_paths (list): list of paths to label files
        train_data_size (string): train100h or train460h or train960h
        map_file_save_path (string): path to mapping files
        is_training (bool, optional): Set True if save as the training set
        is_test (bool, optional): Set True if save as the test set
        frequency_threshold (int, optional): the vocabulary is restricted to
            words which appear more than 'frequency_threshold' in the training
            set
        save_map_file (bool, optional): if True, save the mapping file
        save_path (string, optional): path to save labels.
            If None, don't save labels.
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

                if is_training:
                    # Count word frequency
                    for word in word_list:
                        vocab_set.add(word)
                        if word not in word_count_dict.keys():
                            word_count_dict[word] = 0
                        word_count_dict[word] += 1

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
                                char_capital_set.add(word[i:i + 2])
                            else:
                                char_capital_set.add(word[i])
                        transcript_capital += word

                # Convert space to "_"
                transcript = re.sub(r'\s', SPACE, transcript)

                for c in list(transcript):
                    char_set.add(c)

                speaker_dict[speaker][utt_name] = [
                    transcript, transcript_capital]

                # for debug
                # print(transcript)
                # print(transcript_capital)
                # print('-----')

    char_map_file_path = mkdir_join(
        map_file_save_path, 'character.txt')
    char_capital_map_file_path = mkdir_join(
        map_file_save_path,
        'character_capital_divide_' + train_data_size + '.txt')
    word_map_file_path = mkdir_join(
        map_file_save_path, 'word_' + train_data_size + '_freq' +
        str(frequency_threshold) + '.txt')

    if is_training:
        # Restrict the vocabulary
        oov_list = [word for word, freq in word_count_dict.items()
                    if freq < frequency_threshold]
        original_vocab_num = len(vocab_set)
        vocab_set -= set(oov_list)
    else:
        # Read the mapping file
        with open(word_map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                vocab_set.add(line[0])

    # Reserve some indices
    char_set.discard(SPACE)
    char_set.discard(APOSTROPHE)
    char_capital_set.discard(APOSTROPHE)

    # for debug
    # print(sorted(list(char_set)))
    # print(sorted(list(char_capital_set)))

    if save_map_file:
        # character-level
        if not isfile(char_map_file_path):
            with open(char_map_file_path, 'w') as f:
                char_list = [SPACE] + \
                    sorted(list(char_set)) + [APOSTROPHE, SOS, EOS]
                for i, char in enumerate(char_list):
                    f.write('%s  %s\n' % (char, str(i)))

        # character-level (capital-divided)
        with open(char_capital_map_file_path, 'w') as f:
            char_list = sorted(list(char_capital_set)) + \
                [APOSTROPHE, SOS, EOS]
            for i, char in enumerate(char_list):
                f.write('%s  %s\n' % (char, str(i)))

        # word-level
        vocab_list = sorted(list(vocab_set)) + [OOV, SOS, EOS]
        with open(word_map_file_path, 'w') as f:
            for i, word in enumerate(vocab_list):
                f.write('%s  %s\n' % (word, str(i)))

        print('Original vocab: %d' % original_vocab_num)
        print('Restriced vocab: %d' % (len(vocab_set) + 1))  # + OOV
        total_word_count = np.sum(list(word_count_dict.values()))
        total_oov_word_count = np.sum(
            [count for word, count in word_count_dict.items() if word in oov_list])
        print('OOV rate %f %%' %
              ((total_oov_word_count / total_word_count) * 100))

        char2idx = Char2idx(map_file_path=char_map_file_path)
        char2idx_capital = Char2idx(map_file_path=char_capital_map_file_path)
        word2idx = Word2idx(map_file_path=word_map_file_path)

        if save_path is not None:
            # Save target labels
            print('===> Saving target labels...')
            for speaker, utterance_dict in tqdm(speaker_dict.items()):
                for utt_name, [transcript, transcript_capital] in utterance_dict.items():
                    save_file_name = utt_name + '.npy'

                    if is_test:
                        # Save target labels as string
                        np.save(mkdir_join(save_path, speaker, save_file_name),
                                transcript)
                    else:
                        # Convert to OOV
                        word_list = [
                            word if word in vocab_set else OOV for word in transcript.split(SPACE)]

                        # Convert to index
                        char_index_list = char2idx(
                            transcript, double_letter=False)
                        char_capital_index_list = char2idx_capital(
                            transcript_capital, double_letter=True)
                        word_index_list = word2idx(word_list)

                        # Save target labels as index
                        np.save(mkdir_join(save_path, 'character', speaker, save_file_name),
                                char_index_list)
                        np.save(mkdir_join(save_path, 'character_capital_divide', speaker, save_file_name),
                                char_capital_index_list)
                        np.save(mkdir_join(save_path, 'word_freq' + str(frequency_threshold), speaker, save_file_name),
                                word_index_list)
