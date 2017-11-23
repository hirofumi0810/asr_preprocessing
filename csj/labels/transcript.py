#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels for the End-to-End model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import re
import pandas as pd
from tqdm import tqdm
import jaconv
from collections import OrderedDict

from utils.util import mkdir_join
from csj.labels.fix_trans import fix_transcript
from csj.labels.fix_trans import is_hiragana, is_katakana

# NOTE:
############################################################
# [phone]
# = 36 phones + noise(NZ) = 37 labels
# [phone_divide]
# = 36 phones + noise(NZ), space(_) = 38 labels

# [kana]
# = 145 kana, noise(NZ) = 146 labels
# [kana_divide]
# = 145 kana, noise(NZ), space(_) = 147 labels

# [kanji]
# -subset
# = 2980 kanji, noise(NZ) = 2981 lables
# -fullset
# = 3384 kanji, noise(NZ) = 3385 lables
# [kanji_divide]
# -subset
# = 2980 kanji, noise(NZ), space(_) = 2982 lables
# -fullset
# = 3384 kanji, noise(NZ), space(_) = 3386 lables

# [word]
# -subset
# Original: ? labels + OOV
# -fullset
# Original: ? labels + OOV
############################################################

SPACE = '_'
SIL = 'sil'
NOISE = 'NZ'
OOV = 'OOV'


def read_sdb(label_paths, data_size, vocab_file_save_path, is_test=False,
             save_vocab_file=False, data_type=None):
    """Read transcripts (.sdb) & save files (.npy).
    Args:
        label_paths (list): list of paths to label files
        data_size (string): fullset or subset
        vocab_file_save_path (string): path to vocabulary files
        is_test (bool, optional): Set True if save as the test set
        save_vocab_file (bool, optional): if True, save vocabulary files
        data_type (string, optional): eval1 or eval2 or eval3
    Returns:
        speaker_dict (dict): the dictionary of utterances of each speaker
            key (string) => speaker
            value (dict) => the dictionary of utterance information of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, trans_kanji, trans_kana, trans_phone]
    """
    print('=====> Reading target labels...')

    # Make mapping dictionary from kana to phone
    kana_list = []
    kana2phone_dict = {}
    phone_set = set([])
    with open(join(vocab_file_save_path, '../kana2phone.txt'), 'r') as f:
        for line in f:
            line = line.strip().split('+')
            kana, phone_seq = line
            kana_list.append(kana)
            kana2phone_dict[kana] = phone_seq
            for phone in phone_seq.split(' '):
                phone_set.add(phone)
        kana2phone_dict[SPACE] = SIL
        kana2phone_dict[NOISE] = NOISE

    speaker_dict = OrderedDict()
    char_set = set([])
    word_count_dict = {}
    vocab_set = set([])
    for label_path in tqdm(label_paths):
        col_names = [j for j in range(25)]
        df = pd.read_csv(label_path, names=col_names,
                         encoding='SHIFT-JIS', delimiter='\t', header=None)
        drop_column = col_names
        drop_column.remove(3)
        drop_column.remove(5)
        drop_column.remove(10)
        drop_column.remove(11)
        df = df.drop(drop_column, axis=1)

        utterance_dict = {}
        utt_index_pre = 1
        start_frame_pre, end_frame_pre = None, None
        trans_kana, trans_kanji = '', ''
        speaker = basename(label_path).split('.')[0]
        for key, row in df.iterrows():
            time_info = row[3].split(' ')
            utt_index = int(time_info[0])
            segment = time_info[1].split('-')
            start_frame = int(float(segment[0]) * 100 + 0.5)
            end_frame = int(float(segment[1]) * 100 + 0.5)
            if start_frame_pre is None:
                start_frame_pre = start_frame
            if end_frame_pre is None:
                end_frame_pre = end_frame

            kanji = row[5]  # include kanji characters
            yomi = row[10]
            # pos_tag = row[11]

            # Stack word in the same utterance
            if utt_index == utt_index_pre:
                trans_kana += yomi + ' '
                trans_kanji += kanji + ' '
                utt_index_pre = utt_index
                end_frame_pre = end_frame
                continue
            else:
                # Count the number of kakko
                left_kanji = trans_kanji.count('(')
                right_kanji = trans_kanji.count(')')
                if left_kanji != right_kanji:
                    trans_kana += yomi + ' '
                    trans_kanji += kanji + ' '
                    utt_index_pre = utt_index
                    end_frame_pre = end_frame
                    continue

                left_kana = trans_kana.count('(')
                right_kana = trans_kana.count(')')
                if left_kana != right_kana:
                    trans_kana += yomi + ' '
                    trans_kanji += kanji + ' '
                    utt_index_pre = utt_index
                    end_frame_pre = end_frame
                    continue
                else:
                    # Clean transcript
                    trans_kana = fix_transcript(trans_kana)
                    trans_kanji = fix_transcript(trans_kanji)

                    # Remove double space
                    while '  ' in trans_kana:
                        trans_kana = re.sub(r'[\s]+', ' ', trans_kana)
                    while '  ' in trans_kanji:
                        trans_kanji = re.sub(r'[\s]+', ' ', trans_kanji)

                    # Convert space to "_"
                    trans_kana = re.sub(r'\s', SPACE, trans_kana)
                    trans_kanji = re.sub(r'\s', SPACE, trans_kanji)

                    # Skip silence & noise only utterance
                    if trans_kana.replace(NOISE, '').replace(SPACE, '') != '':

                        # Remove the first and last space
                        if len(trans_kanji) > 0:
                            if trans_kanji[0] == SPACE:
                                trans_kana = trans_kana[1:]
                                trans_kanji = trans_kanji[1:]
                            if trans_kanji[-1] == SPACE:
                                trans_kana = trans_kana[:-1]
                                trans_kanji = trans_kanji[:-1]

                        # for exception
                        if trans_kana[0:2] == 'Z_':
                            trans_kana = trans_kana[2:]

                        for char in list(trans_kanji):
                            char_set.add(char)

                        # Count words
                        word_list = trans_kanji.split(SPACE)
                        for word in word_list:
                            vocab_set.add(word)
                            if word not in word_count_dict.keys():
                                word_count_dict[word] = 0
                            word_count_dict[word] += 1

                        # Convert kana character to phone
                        trans_phone = ' '.join(
                            kana2phone(trans_kana, kana2phone_dict))

                        utterance_dict[str(utt_index - 1).zfill(4)] = [
                            start_frame_pre,
                            end_frame_pre,
                            trans_kanji,
                            trans_kana,
                            trans_phone]

                        # for debug
                        # print(trans_kanji)
                        # print(trans_kana)
                        # print('-----')

                    # Initialization
                    trans_kana = yomi + ' '
                    trans_kanji = kanji + ' '
                    utt_index_pre = utt_index
                    start_frame_pre = start_frame
                    end_frame_pre = end_frame

        # Register all utterances of each speaker
        speaker_dict[speaker] = utterance_dict

    # Make vocabulary files
    word_freq1_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq1_' + data_size + '.txt')
    word_freq5_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq5_' + data_size + '.txt')
    word_freq10_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq10_' + data_size + '.txt')
    word_freq15_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'word_freq15_' + data_size + '.txt')
    char_kanji_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'kanji_' + data_size + '.txt')
    char_kana_vocab_file_path = mkdir_join(vocab_file_save_path, 'kana.txt')
    phone_vocab_file_path = mkdir_join(vocab_file_save_path, 'phone.txt')

    # Remove unneccesary character
    char_set.discard('N')
    char_set.discard('Z')

    # Reserve some indices
    char_set.discard(SPACE)
    vocab_set.discard(NOISE)

    # for debug
    # print(sorted(list(char_set)))

    if save_vocab_file:
        # character-level (kanji)
        kanji_set = set([])
        for char in char_set:
            if (not is_hiragana(char)) and (not is_katakana(char)):
                kanji_set.add(char)
        for kana in kana_list:
            kanji_set.add(kana)
            kanji_set.add(jaconv.kata2hira(kana))
        with open(char_kanji_vocab_file_path, 'w') as f:
            kanji_list = sorted(list(kanji_set)) + [NOISE, SPACE]
            for kanji in kanji_list:
                f.write('%s\n' % kanji)

        # character-level (kana)
        with open(char_kana_vocab_file_path, 'w') as f:
            kana_list_tmp = sorted(kana_list) + [NOISE, SPACE]
            for kana in kana_list_tmp:
                f.write('%s\n' % kana)

        # phone-level
        with open(phone_vocab_file_path, 'w') as f:
            phone_list = sorted(list(phone_set)) + [NOISE, SIL]
            for phone in phone_list:
                f.write('%s\n' % phone)

        # word-level (threshold == 1)
        with open(word_freq1_vocab_file_path, 'w') as f:
            vocab_list = sorted(list(vocab_set)) + [NOISE, OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 5)
        with open(word_freq5_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 5 and word != NOISE]) + [NOISE, OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 10)
        with open(word_freq10_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 10 and word != NOISE]) + [NOISE, OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 15)
        with open(word_freq15_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 15 and word != NOISE]) + [NOISE, OOV]
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


def kana2phone(trans_kana, kana2phone_dict):
    trans_kana_list = list(trans_kana)
    trans_phone_list = []
    i = 0
    while i < len(trans_kana_list):
        # Check whether next character is a double consonant
        if i != len(trans_kana_list) - 1:
            if trans_kana_list[i] + trans_kana_list[i + 1] in kana2phone_dict.keys():
                phone_seq = kana2phone_dict[trans_kana_list[i] +
                                            trans_kana_list[i + 1]]
                trans_phone_list.extend(phone_seq.split(' '))
                i += 1
            elif trans_kana_list[i] in kana2phone_dict.keys():
                trans_phone_list.extend(
                    kana2phone_dict[trans_kana_list[i]].split(' '))
            else:
                raise ValueError(
                    'There are no character such as %s'
                    % trans_kana_list[i])
        else:
            if trans_kana_list[i] in kana2phone_dict.keys():
                trans_phone_list.extend(
                    kana2phone_dict[trans_kana_list[i]].split(' '))
            else:
                raise ValueError(
                    'There are no character such as %s'
                    % trans_kana_list[i])
        i += 1

    return trans_phone_list


def compute_oov_rate(speaker_dict, vocab_file_path):

    with open(vocab_file_path, 'r') as f:
        vocab_set = set([])
        for line in f:
            word = line.strip()
            vocab_set.add(word)

    oov_count = 0
    word_num = 0
    for speaker_dict, utt_dict in speaker_dict.items():
        for utt_name, utt_info in utt_dict.items():
            trans_kanji = utt_info[2]
            word_list = trans_kanji.split(SPACE)
            word_num += len(word_list)

            for word in word_list:
                if word not in vocab_set:
                    oov_count += 1

    oov_rate = oov_count * 100 / word_num

    return oov_rate
