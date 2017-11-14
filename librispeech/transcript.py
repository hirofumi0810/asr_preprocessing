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

### [word, threshold == 1]
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


def read_trans(label_paths, data_size, vocab_file_save_path,
               is_training=False, is_test=False, frequency_threshold=5,
               save_vocab_file=False, save_path=None):
    """Read transcript.
    Args:
        label_paths (list): list of paths to label files
        data_size (string): 100h or 460h or 960h
        vocab_file_save_path (string): path to vocabulary files
        is_training (bool, optional): Set True when proccessing the training set
        is_test (bool, optional): Set True when proccessing the test set
        frequency_threshold (int, optional): the vocabulary is restricted to
            words which appear more than 'frequency_threshold' in the training
            set
        save_vocab_file (bool, optional): if True, save vocabulary files
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

                # Count words
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

    # Make vocabulary files
    char_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'character.txt')
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

    if is_training and save_vocab_file:
        # character-level
        if not isfile(char_vocab_file_path):
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
        original_vocab_num = len(vocab_list) - 1
        print('Original vocab: %d' % original_vocab_num)

        # word-level (threshold == 5)
        with open(word_freq5_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 5]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)
        print('Word (freq5):')
        print('  Restriced vocab: %d' % len(vocab_list))
        print('  OOV rate (train): %f %%' %
              (((original_vocab_num - len(vocab_list) + 1) / original_vocab_num) * 100))

        # word-level (threshold == 10)
        with open(word_freq10_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 10]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)
        print('Word (freq10):')
        print('  Restriced vocab: %d' % len(vocab_list))
        print('  OOV rate (train): %f %%' %
              (((original_vocab_num - len(vocab_list) + 1) / original_vocab_num) * 100))

        # word-level (threshold == 15)
        with open(word_freq15_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 15]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)
        print('Word (freq15):')
        print('  Restriced vocab: %d' % len(vocab_list))
        print('  OOV rate (train): %f %%' %
              (((original_vocab_num - len(vocab_list) + 1) / original_vocab_num) * 100))

    # Compute OOV rate
    if is_test:
        # word-level (threshold == 1)
        with open(word_freq1_vocab_file_path, 'r') as f:
            train_vocab_set = set([])
            for line in f:
                word = line.strip()
                train_vocab_set.add(word)
        oov_count = 0
        for word in vocab_set:
            if word not in train_vocab_set:
                oov_count += 1
        print('Word (freq1):')
        print('  OOV rate (test): %f %%' %
              ((oov_count / len(vocab_set)) * 100))

        # word-level (threshold == 5)
        with open(word_freq5_vocab_file_path, 'r') as f:
            train_vocab_set = set([])
            for line in f:
                word = line.strip()
                train_vocab_set.add(word)
        oov_count = 0
        for word in vocab_set:
            if word not in train_vocab_set:
                oov_count += 1
        print('Word (freq5):')
        print('  OOV rate (test): %f %%' %
              ((oov_count / len(vocab_set)) * 100))

        # word-level (threshold == 10)
        with open(word_freq10_vocab_file_path, 'r') as f:
            train_vocab_set = set([])
            for line in f:
                word = line.strip()
                train_vocab_set.add(word)
        oov_count = 0
        for word in vocab_set:
            if word not in train_vocab_set:
                oov_count += 1
        print('Word (freq10):')
        print('  OOV rate (test): %f %%' %
              ((oov_count / len(vocab_set)) * 100))

        # word-level (threshold == 15)
        with open(word_freq15_vocab_file_path, 'r') as f:
            train_vocab_set = set([])
            for line in f:
                word = line.strip()
                train_vocab_set.add(word)
        oov_count = 0
        for word in vocab_set:
            if word not in train_vocab_set:
                oov_count += 1
        print('Word (freq15):')
        print('  OOV rate (test): %f %%' %
              ((oov_count / len(vocab_set)) * 100))

    if not is_test:
        char2idx = Char2idx(char_vocab_file_path)
        char2idx_capital = Char2idx(
            char_capital_vocab_file_path,
            double_letter=True)
        word2idx_freq1 = Word2idx(word_freq1_vocab_file_path)
        word2idx_freq5 = Word2idx(word_freq5_vocab_file_path)
        word2idx_freq10 = Word2idx(word_freq10_vocab_file_path)
        word2idx_freq15 = Word2idx(word_freq15_vocab_file_path)

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for speaker, utterance_dict in tqdm(speaker_dict.items()):
            for utt_name, [transcript, transcript_capital] in utterance_dict.items():
                save_file_name = utt_name + '.npy'
                # ex.) utt_name: speaker-book-utt_index

                if is_test:
                    # Save target labels as string
                    np.save(mkdir_join(save_path, 'character',
                                       speaker, save_file_name), transcript)
                    np.save(mkdir_join(save_path, 'character_capital_divide',
                                       speaker, save_file_name), transcript)
                    np.save(mkdir_join(save_path, 'word_freq1',
                                       speaker, save_file_name), transcript)
                    np.save(mkdir_join(save_path, 'word_freq5',
                                       speaker, save_file_name), transcript)
                    np.save(mkdir_join(save_path, 'word_freq10',
                                       speaker, save_file_name), transcript)
                    np.save(mkdir_join(save_path, 'word_freq15',
                                       speaker, save_file_name), transcript)
                else:
                    # Save target labels as index
                    np.save(mkdir_join(save_path, 'character', speaker, save_file_name),
                            char2idx(transcript))
                    np.save(mkdir_join(save_path, 'character_capital_divide', speaker, save_file_name),
                            char2idx_capital(transcript_capital))
                    np.save(mkdir_join(save_path, 'word_freq1', speaker, save_file_name),
                            word2idx_freq1(word_list))
                    np.save(mkdir_join(save_path, 'word_freq5', speaker, save_file_name),
                            word2idx_freq5(word_list))
                    np.save(mkdir_join(save_path, 'word_freq10', speaker, save_file_name),
                            word2idx_freq10(word_list))
                    np.save(mkdir_join(save_path, 'word_freq15', speaker, save_file_name),
                            word2idx_freq15(word_list))
