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

from utils.labels.phone import Phone2idx
from utils.labels.character import Char2idx
from utils.labels.word import Word2idx
from utils.util import mkdir_join
from csj.labels.fix_trans import fix_transcript
from csj.labels.fix_trans import is_hiragana, is_katakana

SPACE = '_'
SIL = 'sil'
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
                value (list) => [start_frame, end_frame,
                                kanji_indices, kanji_div_indices,
                                kana_indices, kana_div_indices,
                                phone_indices, phone_div_indices,
                                word_freq1_indices, word_freq5_indices,
                                word_freq10_indices, word_freq15_indices]
    """
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

    print('=====> Reading target labels...')
    speaker_dict = OrderedDict()
    char_set = set([])
    word_count_dict = {}
    vocab_set = set([])
    for label_path in tqdm(label_paths):
        col_names = [j for j in range(25)]
        df = pd.read_csv(label_path, names=col_names,
                         encoding='SHIFT-JIS', delimiter='\t', header=None)

        utt_dict = OrderedDict()
        utt_index_pre = 1
        start_frame_pre, end_frame_pre = None, None
        trans_kana, trans_kanji, trans_pos = '', '', ''
        speaker = basename(label_path).split('.')[0]
        for key, row in df.iterrows():

            # From kaldi
            time = row[3]  # Time information for segment
            word = row[5]  # Word
            # num = row[9]  # Number and point
            # About morpheme
            if isinstance(row[11], str):
                pos = row[11]  # Part Of Speech
            else:
                pos = ''
            # acf = row[12]  # A Conjugated Form
            # kacf = row[13]  # Kind of A Conjugated Form
            # kav = row[14]  # Kind of Aulxiliary Verb
            # ec = row[15]  # Euphonic Change
            # other = row[16]  # Other information
            pron = row[10]  # Pronunciation for lexicon

            utt_index = int(time.split(' ')[0])
            segment = time.split(' ')[1].split('-')
            start_frame = int(float(segment[0]) * 100 + 0.5)
            end_frame = int(float(segment[1]) * 100 + 0.5)
            if start_frame_pre is None:
                start_frame_pre = start_frame
            if end_frame_pre is None:
                end_frame_pre = end_frame

            # Stack word in the same utterance
            if utt_index == utt_index_pre:
                trans_kanji += word + ' '
                trans_kana += pron + ' '
                if pos != '':
                    trans_pos += pos + ' '
                utt_index_pre = utt_index
                end_frame_pre = end_frame
                continue

            # Count the number of brackets
            if trans_kanji.count('(') != trans_kanji.count(')'):
                trans_kanji += word + ' '
                trans_kana += pron + ' '
                if pos != '':
                    trans_pos += pos + ' '
                utt_index_pre = utt_index
                end_frame_pre = end_frame
                continue

            if trans_kana.count('(') != trans_kana.count(')'):
                trans_kanji += word + ' '
                trans_kana += pron + ' '
                if pos != '':
                    trans_pos += pos + ' '
                utt_index_pre = utt_index
                end_frame_pre = end_frame
                continue

            # if '<P:' in trans_kana:
            #     print(label_path)
            #     print(trans_kanji)
            #     print(trans_kana)

            # Clean transcript
            trans_kanji = fix_transcript(trans_kanji)
            trans_kana = fix_transcript(trans_kana)

            # Remove double space
            while '  ' in trans_kanji:
                trans_kanji = re.sub(r'[\s]+', ' ', trans_kanji)
            while '  ' in trans_kana:
                trans_kana = re.sub(r'[\s]+', ' ', trans_kana)
            while '  ' in trans_pos:
                trans_pos = re.sub(r'[\s]+', ' ', trans_pos)

            # Skip silence only utterance
            if trans_kanji.replace(' ', '') != '' and len(trans_pos) > 0:

                # Remove the first and last space
                if len(trans_kanji) > 0 and trans_kanji[0] == ' ':
                    trans_kanji = trans_kanji[1:]
                if len(trans_kana) > 0 and trans_kana[0] == ' ':
                    trans_kana = trans_kana[1:]
                if len(trans_kanji) > 0 and trans_kanji[-1] == ' ':
                    trans_kanji = trans_kanji[:-1]
                if len(trans_kana) > 0 and trans_kana[-1] == ' ':
                    trans_kana = trans_kana[:-1]

                # Convert space to "_"
                trans_kanji = re.sub(r'\s', SPACE, trans_kanji)
                trans_kana = re.sub(r'\s', SPACE, trans_kana)

                # For exception
                if trans_kana[0:2] == 'Z_':
                    trans_kana = trans_kana[2:]

                for c in list(trans_kanji):
                    char_set.add(c)

                # Count words
                word_list = trans_kanji.split(SPACE)
                for w in word_list:
                    vocab_set.add(w)
                    if w not in word_count_dict.keys():
                        word_count_dict[w] = 0
                    word_count_dict[w] += 1

                # Convert kana character to phone
                trans_phone = ' '.join(
                    kana2phone(trans_kana, kana2phone_dict))

                utt_dict[str(utt_index - 1).zfill(4)] = [
                    start_frame_pre, end_frame_pre,
                    trans_kanji, trans_kana, trans_phone]

                # for debug
                # print(trans_kanji)
                # print(trans_kana)
                # print(trans_phone)
                # print('-----')

            # Initialization
            trans_kanji = word + ' '
            trans_kana = pron + ' '
            if pos == '':
                trans_pos = ''
            else:
                trans_pos = pos + ' '
            utt_index_pre = utt_index
            start_frame_pre = start_frame
            end_frame_pre = end_frame

        # Register all utterances of each speaker
        speaker_dict[speaker] = utt_dict

    # Make vocabulary files
    kanji_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'kanji_' + data_size + '.txt')
    kanji_div_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'kanji_divide_' + data_size + '.txt')
    kana_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'kana_' + data_size + '.txt')
    kana_div_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'kana_divide_' + data_size + '.txt')
    phone_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'phone_' + data_size + '.txt')
    phone_div_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'phone_divide_' + data_size + '.txt')
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

    # for debug
    # print(sorted(list(char_set)))

    if save_vocab_file:
        # character-level (kanji, kanji_divide)
        kanji_set = set([])
        for char in char_set:
            if (not is_hiragana(char)) and (not is_katakana(char)):
                kanji_set.add(char)
        for kana in kana_list:
            kanji_set.add(kana)
            kanji_set.add(jaconv.kata2hira(kana))
        with open(kanji_vocab_file_path, 'w') as f, open(kanji_div_vocab_file_path, 'w') as f_div:
            kanji_list = sorted(list(kanji_set))
            for kanji in kanji_list:
                f.write('%s\n' % kanji)
            for kanji in kanji_list + [SPACE]:
                f_div.write('%s\n' % kanji)

        # character-level (kana, kana_divide)
        with open(kana_vocab_file_path, 'w') as f, open(kana_div_vocab_file_path, 'w') as f_div:
            kana_list_tmp = sorted(kana_list)
            for kana in kana_list_tmp:
                f.write('%s\n' % kana)
            for kana in kana_list_tmp + [SPACE]:
                f_div.write('%s\n' % kana)

        # phone-level (phone, phone_divide)
        with open(phone_vocab_file_path, 'w') as f, open(phone_div_vocab_file_path, 'w') as f_div:
            phone_list = sorted(list(phone_set))
            for phone in phone_list:
                f.write('%s\n' % phone)
            for phone in phone_list + [SIL]:
                f_div.write('%s\n' % phone)

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

    # Tokenize
    print('=====> Tokenize...')
    kanji2idx = Char2idx(kanji_vocab_file_path, double_letter=True)
    kanji2idx_div = Char2idx(kanji_div_vocab_file_path, double_letter=True)
    kana2idx = Char2idx(kana_vocab_file_path, double_letter=True)
    kana2idx_div = Char2idx(kana_div_vocab_file_path, double_letter=True)
    phone2idx = Phone2idx(phone_vocab_file_path)
    phone2idx_div = Phone2idx(phone_div_vocab_file_path)
    word2idx_freq1 = Word2idx(word_freq1_vocab_file_path)
    word2idx_freq5 = Word2idx(word_freq5_vocab_file_path)
    word2idx_freq10 = Word2idx(word_freq10_vocab_file_path)
    word2idx_freq15 = Word2idx(word_freq15_vocab_file_path)
    for speaker, utt_dict in tqdm(speaker_dict.items()):
        for utt_index, utt_info in utt_dict.items():
            start_frame, end_frame, trans_kanji, trans_kana, trans_phone = utt_info
            if is_test:
                utt_dict[utt_index] = [
                    start_frame, end_frame,
                    trans_kanji.replace(SPACE, ''), trans_kanji,
                    trans_kana.replace(SPACE, ''), trans_kana,
                    trans_phone.replace(SIL, '').replace(
                        '  ', ' '), trans_phone,
                    trans_kanji, trans_kanji, trans_kanji, trans_kanji]
            else:
                kanji_indices = kanji2idx(trans_kanji.replace(SPACE, ''))
                kanji_div_indices = kanji2idx_div(trans_kanji)
                kana_indices = kana2idx(trans_kana.replace(SPACE, ''))
                kana_div_indices = kana2idx_div(trans_kana)
                phone_indices = phone2idx(
                    trans_phone.replace(SIL, '').replace('  ', ' '))
                phone_div_indices = phone2idx_div(trans_phone)
                word_freq1_indices = word2idx_freq1(trans_kanji)
                word_freq5_indices = word2idx_freq5(trans_kanji)
                word_freq10_indices = word2idx_freq10(trans_kanji)
                word_freq15_indices = word2idx_freq15(trans_kanji)

                kanji_indices = int2str(kanji_indices)
                kanji_div_indices = int2str(kanji_div_indices)
                kana_indices = int2str(kana_indices)
                kana_div_indices = int2str(kana_div_indices)
                phone_indices = int2str(phone_indices)
                phone_div_indices = int2str(phone_div_indices)
                word_freq1_indices = int2str(word_freq1_indices)
                word_freq5_indices = int2str(word_freq5_indices)
                word_freq10_indices = int2str(word_freq10_indices)
                word_freq15_indices = int2str(word_freq15_indices)

                utt_dict[utt_index] = [
                    start_frame, end_frame,
                    kanji_indices, kanji_div_indices,
                    kana_indices, kana_div_indices,
                    phone_indices, phone_div_indices,
                    word_freq1_indices, word_freq5_indices,
                    word_freq10_indices, word_freq15_indices]

        speaker_dict[speaker] = utt_dict

    return speaker_dict


def int2str(indices):
    return' '.join(list(map(str, indices.tolist())))


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
