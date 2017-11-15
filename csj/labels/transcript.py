#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels for the End-to-End model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import jaconv

from utils.util import mkdir_join
from utils.labels.word import Word2idx
from utils.labels.character import Char2idx
from utils.labels.phone import Phone2idx
from csj.labels.fix_trans import fix_transcript
from csj.labels.fix_trans import is_hiragana, is_katakana

# NOTE:
############################################################
# [phone]
# = 36 + noise(NZ), space(_) = 38 labels

# [kana]
# = 145 kana, noise(NZ) = 148 labels
# [kana_divide]
# = 145 kana, noise(NZ), space(_) = 147 labels

# [kanji, subset]
# = 2980 kanji, noise(NZ) = 2983 lables
# [kanji_divide, subset]
# = 2980 kanji, noise(NZ), space(_) = 2982 lables

# [kanji, fullset]
# = 3384 kanji, noise(NZ) = 3387 lables
# [kanji_divide, fullset]
# = 3384 kanji, noise(NZ), space(_) = 3386 lables

# [word, subset, threshold == 1]
# Original:  labels

# [word, subset, threshold == 1]
# Original:  labels
############################################################

SPACE = '_'
SIL = 'sil'
NOISE = 'NZ'
OOV = 'OOV'


def read_sdb(label_paths, data_size, vocab_file_save_path,
             is_training=False, is_test=False,
             save_vocab_file=False, save_path=None):
    """Read transcripts (.sdb) & save files (.npy).
    Args:
        label_paths (list): list of paths to label files
        data_size (string): fullset or subset
        vocab_file_save_path (string): path to vocabulary files
        is_training (bool, optional): Set True if save as the training set
        is_test (bool, optional): Set True if save as the test set
        save_vocab_file (bool, optional): if True, save vocabulary files
        save_path (string, optional): path to save labels.
            If None, don't save labels.
    Returns:
        speaker_dict (dict): the dictionary of utterances of each speaker
            key (string) => speaker
            value (dict) => the dictionary of utterance information of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, trans_kana, trans_kanji]
    """
    print('===> Reading target labels...')
    speaker_dict = {}
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

                        # Skip long utterances (longer than 20s)
                        duration = (end_frame_pre - start_frame_pre) / 100
                        if duration < 20:

                            utterance_dict[str(utt_index - 1).zfill(4)] = [
                                start_frame_pre,
                                end_frame_pre,
                                trans_kana,
                                trans_kanji]

                            for char in list(trans_kanji):
                                char_set.add(char)

                            # Count words
                            word_list = trans_kanji.split(SPACE)
                            for word in word_list:
                                vocab_set.add(word)
                                if word not in word_count_dict.keys():
                                    word_count_dict[word] = 0
                                word_count_dict[word] += 1

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

    if is_training and save_vocab_file:
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
        original_vocab_num = len(vocab_list) - 1
        print('Original vocab: %d' % original_vocab_num)

        # word-level (threshold == 5)
        with open(word_freq5_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 5 and word != NOISE]) + [NOISE, OOV]
            for word in vocab_list:
                f.write('%s\n' % word)
        print('Word (freq5):')
        print('  Restriced vocab: %d' % len(vocab_list))
        print('  OOV rate (train): %f %%' %
              (((original_vocab_num - len(vocab_list) + 1) / original_vocab_num) * 100))

        # word-level (threshold == 10)
        with open(word_freq10_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 10 and word != NOISE]) + [NOISE, OOV]
            for word in vocab_list:
                f.write('%s\n' % word)
        print('Word (freq10):')
        print('  Restriced vocab: %d' % len(vocab_list))
        print('  OOV rate (train): %f %%' %
              (((original_vocab_num - len(vocab_list) + 1) / original_vocab_num) * 100))

        # word-level (threshold == 15)
        with open(word_freq15_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 15 and word != NOISE]) + [NOISE, OOV]
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
        kanji2idx = Char2idx(char_kanji_vocab_file_path,
                             double_letter=True,
                             remove_list=[SPACE])
        kana2idx = Char2idx(char_kana_vocab_file_path,
                            double_letter=True,
                            remove_list=[SPACE])
        phone2idx = Phone2idx(phone_vocab_file_path,
                              remove_list=[SIL])
        kanji2idx_divide = Char2idx(char_kanji_vocab_file_path,
                                    double_letter=True)
        kana2idx_divide = Char2idx(char_kana_vocab_file_path,
                                   double_letter=True)
        phone2idx_divide = Phone2idx(phone_vocab_file_path)
        word2idx_freq1 = Word2idx(word_freq1_vocab_file_path)
        word2idx_freq5 = Word2idx(word_freq5_vocab_file_path)
        word2idx_freq10 = Word2idx(word_freq10_vocab_file_path)
        word2idx_freq15 = Word2idx(word_freq15_vocab_file_path)

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for speaker, utterance_dict in tqdm(speaker_dict.items()):
            for utt_index, utt_info in utterance_dict.items():
                start_frame, end_frame, trans_kana, trans_kanji = utt_info
                save_file_name = speaker + '_' + utt_index + '.npy'

                # kanji & kana characters
                if is_test:
                    # Save target labels as string
                    np.save(mkdir_join(save_path, 'kanji', speaker, save_file_name),
                            trans_kanji.replace(SPACE, ''))
                    np.save(mkdir_join(save_path, 'kana', speaker, save_file_name),
                            trans_kana.replace(SPACE, ''))
                    np.save(mkdir_join(save_path, 'kanji_divide', speaker, save_file_name),
                            trans_kanji.replace(SPACE, ''))
                    np.save(mkdir_join(save_path, 'kana_divide', speaker, save_file_name),
                            trans_kana.replace(SPACE, ''))
                    np.save(mkdir_join(save_path, 'word_freq1', speaker, save_file_name),
                            trans_kanji)
                    np.save(mkdir_join(save_path, 'word_freq5', speaker, save_file_name),
                            trans_kanji)
                    np.save(mkdir_join(save_path, 'word_freq10', speaker, save_file_name),
                            trans_kanji)
                    np.save(mkdir_join(save_path, 'word_freq15', speaker, save_file_name),
                            trans_kanji)
                else:
                    word_list = trans_kanji.split(SPACE)

                    # Save target labels as index
                    np.save(mkdir_join(save_path, 'kanji', speaker, save_file_name),
                            kanji2idx(trans_kanji.replace(SPACE, '')))
                    np.save(mkdir_join(save_path, 'kana', speaker, save_file_name),
                            kana2idx(trans_kana.replace(SPACE, '')))
                    np.save(mkdir_join(save_path, 'kanji_divide', speaker, save_file_name),
                            kanji2idx_divide(trans_kanji))
                    np.save(mkdir_join(save_path, 'kana_divide', speaker, save_file_name),
                            kana2idx_divide(trans_kana))
                    np.save(mkdir_join(save_path, 'word_freq1', speaker, save_file_name),
                            word2idx_freq1(word_list))
                    np.save(mkdir_join(save_path, 'word_freq5', speaker, save_file_name),
                            word2idx_freq5(word_list))
                    np.save(mkdir_join(save_path, 'word_freq10', speaker, save_file_name),
                            word2idx_freq10(word_list))
                    np.save(mkdir_join(save_path, 'word_freq15', speaker, save_file_name),
                            word2idx_freq15(word_list))

                # Convert kana character to phone
                trans_phone_list = kana2phone(
                    trans_kana.replace(SPACE, ''), kana2phone_dict)
                trans_phone_divide_list = kana2phone(
                    trans_kana, kana2phone_dict)

                # phone
                if is_test:
                    # Save target labels as string
                    np.save(mkdir_join(save_path, 'phone', speaker, save_file_name),
                            ' '.join(trans_phone_list))
                    np.save(mkdir_join(save_path, 'phone_divide', speaker, save_file_name),
                            ' '.join(trans_phone_divide_list))
                else:
                    # Save target labels as index
                    np.save(mkdir_join(save_path, 'phone', speaker, save_file_name),
                            phone2idx(trans_phone_list))
                    np.save(mkdir_join(save_path, 'phone_divide', speaker, save_file_name),
                            phone2idx_divide(trans_phone_divide_list))

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
