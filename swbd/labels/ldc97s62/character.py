#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels for the End-to-End model (LDC97S62 corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm
from collections import OrderedDict

from swbd.labels.ldc97s62.word_boundary import read_segmentation
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


def read_trans(label_paths, word_boundary_paths, run_root_path,
               vocab_file_save_path,
               save_vocab_file=False,  speaker_dict_fisher=None,
               char_set=None, char_capital_set=None, word_count_dict=None):
    """Read transcripts (*_trans.txt) & save files (.npy).
    Args:
        label_paths (list): list of paths to label files
        word_boundary_paths (list): list of paths to word boundary files
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

    for label_path, wb_path in zip(tqdm(label_paths), word_boundary_paths):
        assert label_path == wb_path.replace('word', 'trans')
        utterance_dict = OrderedDict()
        segmentation_dict = read_segmentation(wb_path)
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

                if transcript == '[silence]':
                    continue

                # Divide into short utterances
                length_threshold = 700
                if end_frame - start_frame >= length_threshold:
                    word_info_list = segmentation_dict[utt_index]
                    divide_points = []
                    divided_trans = []
                    partial_word_list = []
                    start_frame_tmp = start_frame
                    for i, word_info in enumerate(word_info_list):
                        if word_info[2] != '':
                            partial_word_list.append(word_info[2])
                        if 0 < i < len(word_info_list) - 1 and word_info[2] == '' and word_info[1] - start_frame_tmp >= length_threshold:
                            divide_points.append(
                                int((word_info[1] + word_info[0]) / 2))
                            divided_trans.append(' '.join(partial_word_list))
                            partial_word_list = []
                            start_frame_tmp = word_info[0]

                    # Last segment
                    if len(partial_word_list) > 0:
                        divided_trans.append(' '.join(partial_word_list))

                    if len(divide_points) > 0:
                        transcript_list = divided_trans
                    else:
                        transcript_list = [transcript]
                else:
                    divide_points = []
                    transcript_list = [transcript]

                for i_trans, trans in enumerate(transcript_list):
                    # Clean transcript
                    trans = fix_transcript(trans)

                    # Convert space to "_"
                    trans = re.sub(r'\s', SPACE, trans)

                    # Skip silence, laughter, noise, vocalized-noise
                    if trans.replace(NOISE, '').replace(LAUGHTER, '').replace(VOCALIZED_NOISE, '').replace(SPACE, '') == '':
                        continue

                    # Remove the first and last space
                    if trans[0] == SPACE:
                        trans = trans[1:]
                    if trans[-1] == SPACE:
                        trans = trans[:-1]

                    # Count words
                    for word in trans.split(SPACE):
                        vocab_set.add(word)
                        if word not in word_count_dict.keys():
                            word_count_dict[word] = 0
                        word_count_dict[word] += 1

                    # Capital-divided
                    trans_capital = ''
                    for word in trans.split(SPACE):
                        if len(word) == 1:
                            char_capital_set.add(word)
                            trans_capital += word
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
                            trans_capital += word

                    for c in list(trans):
                        char_set.add(c)

                    if len(transcript_list) == 1:
                        utterance_dict[utt_index.zfill(4)] = [
                            start_frame, end_frame, trans]
                    else:
                        assert len(transcript_list) - 1 == len(divide_points)
                        if i_trans == 0:
                            assert start_frame < divide_points[i_trans] - 1
                            utterance_dict[utt_index.zfill(4) + '-' + str(i_trans + 1)] = [
                                start_frame, divide_points[0] - 1, trans]
                        elif i_trans == len(transcript_list) - 1:
                            assert start_frame < end_frame
                            utterance_dict[utt_index.zfill(4) + '-' + str(i_trans + 1)] = [
                                divide_points[-1], end_frame, trans]
                        else:
                            assert divide_points[i_trans -
                                                 1] < divide_points[i_trans] - 1
                            utterance_dict[utt_index.zfill(4) + '-' + str(i_trans + 1)] = [
                                divide_points[i_trans - 1], divide_points[i_trans] - 1, trans]

                    # for debug
                    # print(transcript_original)
                    # print(trans)
                    # print(trans_capital)

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
