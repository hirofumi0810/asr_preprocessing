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
from utils.labels.character import Char2idx
from utils.labels.phone import Phone2idx
from csj.labels.fix_trans import fix_transcript
from csj.labels.fix_trans import is_hiragana, is_katakana

# NOTE:
############################################################
# [phone]
# = 36 + noise(NZ), space(_), <SOS>, <EOS> = 40 labels

# [kana]
# = 145 kana, noise(NZ), <SOS>, <EOS> = 148 labels
# [kana_divide]
# = 145 kana, noise(NZ), space(_), <SOS>, <EOS> = 149 labels

# [kanji, train_subset]
# = 2980 kanji, noise(NZ), <SOS>, <EOS> = 2983 lables
# [kanji_divide, train_subset]
# = 2980 kanji, noise(NZ), space(_), <SOS>, <EOS> = 2984 lables

# [kanji, train_fullset]
# = 3384 kanji, noise(NZ), <SOS>, <EOS> = 3387 lables
# [kanji_divide, train_fullset]
# = 3384 kanji, noise(NZ), space(_), <SOS>, <EOS> = 3388 lables
############################################################

SPACE = '_'
SIL = 'sil'
NOISE = 'NZ'
SOS = '<'
EOS = '>'


def read_sdb(label_paths, train_data_size, map_file_save_path,
             is_training=False, is_test=False,
             save_map_file=False, save_path=None):
    """Read transcripts (.sdb) & save files (.npy).
    Args:
        label_paths (list): list of paths to label files
        train_data_size (string): train_fullset or train_subset
        map_file_save_path (string): path to mapping files
        is_training (bool, optional): Set True if save as the training set
        is_test (bool, optional): Set True if save as the test set
        save_map_file (bool, optional): if True, save the mapping file
        save_path (string, optional): path to save labels.
            If None, don't save labels.
    Returns:
        speaker_dict (dict): the dictionary of utterances of each speaker
            key (string) => speaker
            value (dict) => the dictionary of utterance information of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, trans_kana, trans_kanji, trans_kana_divide, trans_kanji_divide]
    """
    print('===> Reading target labels...')
    speaker_dict = {}
    kanji_char_set = set([])
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
        trans_kana_divide, trans_kanji_divide = '', ''
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
                trans_kana += yomi
                trans_kanji += kanji
                trans_kana_divide += yomi + ' '
                trans_kanji_divide += kanji + ' '

                utt_index_pre = utt_index
                end_frame_pre = end_frame
                continue
            else:
                # Count the number of kakko
                left_kanji = trans_kanji.count('(')
                right_kanji = trans_kanji.count(')')
                if left_kanji != right_kanji:
                    trans_kana += yomi
                    trans_kanji += kanji
                    trans_kana_divide += yomi + ' '
                    trans_kanji_divide += kanji + ' '

                    utt_index_pre = utt_index
                    end_frame_pre = end_frame
                    continue

                left_kana = trans_kana.count('(')
                right_kana = trans_kana.count(')')
                if left_kana != right_kana:
                    trans_kana += yomi
                    trans_kanji += kanji
                    trans_kana_divide += yomi + ' '
                    trans_kanji_divide += kanji + ' '

                    utt_index_pre = utt_index
                    end_frame_pre = end_frame
                    continue
                else:
                    # Clean transcript
                    trans_kana = fix_transcript(trans_kana)
                    trans_kanji = fix_transcript(trans_kanji)
                    trans_kana_divide = fix_transcript(trans_kana_divide)
                    trans_kanji_divide = fix_transcript(trans_kanji_divide)

                    # Remove double space
                    while '  ' in trans_kana:
                        trans_kana = re.sub(r'[\s]+', ' ', trans_kana)
                    while '  ' in trans_kanji:
                        trans_kanji = re.sub(r'[\s]+', ' ', trans_kanji)
                    while '  ' in trans_kana_divide:
                        trans_kana_divide = re.sub(
                            r'[\s]+', ' ', trans_kana_divide)
                    while '  ' in trans_kanji_divide:
                        trans_kanji_divide = re.sub(
                            r'[\s]+', ' ', trans_kanji_divide)

                    # Convert space to "_"
                    trans_kana = re.sub(r'\s', SPACE, trans_kana)
                    trans_kanji = re.sub(r'\s', SPACE, trans_kanji)
                    trans_kana_divide = re.sub(
                        r'\s', SPACE, trans_kana_divide)
                    trans_kanji_divide = re.sub(
                        r'\s', SPACE, trans_kanji_divide)

                    # Skip silence & noise only utterance
                    if trans_kana.replace(NOISE, '').replace(SPACE, '') != '':
                        for char in list(trans_kanji):
                            kanji_char_set.add(char)

                        utterance_dict[str(utt_index - 1).zfill(4)] = [
                            start_frame_pre,
                            end_frame_pre,
                            trans_kana,
                            trans_kanji,
                            trans_kana_divide,
                            trans_kanji_divide]

                        # for debug
                        # print(trans_kanji)
                        # print(trans_kanji_divide)
                        # print(trans_kana)
                        # print(trans_kana_divide)
                        # print('-----')

                    # Initialization
                    trans_kana = yomi
                    trans_kanji = kanji
                    trans_kana_divide = yomi + ' '
                    trans_kanji_divide = kanji + ' '

                    utt_index_pre = utt_index
                    start_frame_pre = start_frame
                    end_frame_pre = end_frame

        # Register all utterances of each speaker
        speaker_dict[speaker] = utterance_dict

    # Make mapping dictionary from kana to phone
    kana_list = []
    kana2phone_dict = {}
    phone_set = set([])
    with open(join(map_file_save_path, '../kana2phone.txt'), 'r') as f:
        for line in f:
            line = line.strip().split('+')
            kana, phone_seq = line
            kana_list.append(kana)
            kana2phone_dict[kana] = phone_seq
            for phone in phone_seq.split(' '):
                phone_set.add(phone)
        kana2phone_dict[SPACE] = SIL
        kana2phone_dict[NOISE] = NOISE
        kana2phone_dict[SOS] = SOS
        kana2phone_dict[EOS] = EOS

    # Make the mapping file (from kanji, kana, phone to number)
    kanji_map_file_path = mkdir_join(
        map_file_save_path, 'kanji_' + train_data_size + '.txt')
    kana_map_file_path = mkdir_join(map_file_save_path, 'kana.txt')
    phone_map_file_path = mkdir_join(map_file_save_path, 'phone.txt')
    kanji_divide_map_file_path = mkdir_join(
        map_file_save_path, 'kanji_divide_' + train_data_size + '.txt')
    kana_divide_map_file_path = mkdir_join(
        map_file_save_path, 'kana_divide.txt')
    phone_divide_map_file_path = mkdir_join(
        map_file_save_path, 'phone_divide.txt')

    # Remove unneccesary charaacter
    kanji_char_set.discard('N')
    kanji_char_set.discard('Z')

    # Reserve some indices
    kanji_char_set.discard(SPACE)
    kanji_char_set.discard(SOS)
    kanji_char_set.discard(EOS)

    # for debug
    # print(sorted(list(kanji_char_set)))

    if save_map_file:
        # kanji-level
        kanji_set = set([])
        for char in kanji_char_set:
            if (not is_hiragana(char)) and (not is_katakana(char)):
                kanji_set.add(char)
        for kana in kana_list:
            kanji_set.add(kana)
            kanji_set.add(jaconv.kata2hira(kana))
        with open(kanji_map_file_path, 'w') as f:
            kanji_list = sorted(list(kanji_set)) + [NOISE, SOS, EOS]
            for i, kanji in enumerate(kanji_list):
                f.write('%s  %s\n' % (kanji, str(i)))
        with open(kanji_divide_map_file_path, 'w') as f:
            kanji_divide_list = sorted(
                list(kanji_set)) + [NOISE, SPACE, SOS, EOS]
            for i, kanji in enumerate(kanji_divide_list):
                f.write('%s  %s\n' % (kanji, str(i)))

        # kana-level
        with open(kana_map_file_path, 'w') as f:
            kana_list_tmp = kana_list + [NOISE, SOS, EOS]
            for i, kana in enumerate(kana_list_tmp):
                f.write('%s  %s\n' % (kana, str(i)))
        with open(kana_divide_map_file_path, 'w') as f:
            kana_divide_list = kana_list + [NOISE, SPACE, SOS, EOS]
            for i, kana in enumerate(kana_divide_list):
                f.write('%s  %s\n' % (kana, str(i)))

        # phone-level
        with open(phone_map_file_path, 'w') as f:
            phone_list = sorted(list(phone_set)) + [NOISE, SOS, EOS]
            for i, phone in enumerate(phone_list):
                f.write('%s  %s\n' % (phone, str(i)))
        with open(phone_divide_map_file_path, 'w') as f:
            phone_divide_list = sorted(
                list(phone_set)) + [NOISE, SIL, SOS, EOS]
            for i, phone in enumerate(phone_divide_list):
                f.write('%s  %s\n' % (phone, str(i)))

    kanji2idx = Char2idx(map_file_path=kanji_map_file_path,
                         double_letter=True)
    kana2idx = Char2idx(map_file_path=kana_map_file_path,
                        double_letter=True)
    phone2idx = Phone2idx(map_file_path=phone_map_file_path)
    kanji2idx_divide = Char2idx(map_file_path=kanji_divide_map_file_path,
                                double_letter=True)
    kana2idx_divide = Char2idx(map_file_path=kana_divide_map_file_path,
                               double_letter=True)
    phone2idx_divide = Phone2idx(map_file_path=phone_divide_map_file_path)

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for speaker, utterance_dict in tqdm(speaker_dict.items()):
            for utt_index, utt_info in utterance_dict.items():
                start_frame, end_frame, trans_kana, trans_kanji, trans_kana_divide, trans_kanji_divide = utt_info
                save_file_name = speaker + '_' + utt_index + '.npy'

                # kanji & kanji
                if is_test:
                    # Save target labels as string
                    np.save(mkdir_join(save_path, 'kanji', speaker, save_file_name),
                            trans_kanji)
                    np.save(mkdir_join(save_path, 'kana', speaker, save_file_name),
                            trans_kana)
                    np.save(mkdir_join(save_path, 'kanji_divide', speaker, save_file_name),
                            trans_kanji_divide)
                    np.save(mkdir_join(save_path, 'kana_divide', speaker, save_file_name),
                            trans_kana_divide)
                    # NOTE: save test transcripts as stirng rather than index
                else:
                    # Convert to index
                    kanji_index_list = kanji2idx(trans_kanji)
                    kana_index_list = kana2idx(trans_kana)
                    kanji_divide_index_list = kanji2idx_divide(
                        trans_kanji_divide)
                    kana_divide_index_list = kana2idx_divide(trans_kana_divide)

                    # Save target labels as index
                    np.save(mkdir_join(save_path, 'kanji', speaker, save_file_name),
                            kanji_index_list)
                    np.save(mkdir_join(save_path, 'kana', speaker, save_file_name),
                            kana_index_list)
                    np.save(mkdir_join(save_path, 'kanji_divide', speaker, save_file_name),
                            kanji_divide_index_list)
                    np.save(mkdir_join(save_path, 'kana_divide', speaker, save_file_name),
                            kana_divide_index_list)

                # Convert kana character to phone
                trans_phone_list = kana2phone(trans_kana, kana2phone_dict)
                trans_phone_divide_list = kana2phone(
                    trans_kana_divide, kana2phone_dict)

                # phone
                if is_test:
                    # Save target labels as string
                    np.save(mkdir_join(save_path, 'phone', speaker, save_file_name),
                            ' '.join(trans_phone_list))
                    np.save(mkdir_join(save_path, 'phone_divide', speaker, save_file_name),
                            ' '.join(trans_phone_divide_list))
                else:
                    # Convert from phone to index
                    phone_index_list = phone2idx(trans_phone_list)
                    phone_divide_index_list = phone2idx_divide(
                        trans_phone_divide_list)

                    # Save target labels as index
                    np.save(mkdir_join(save_path, 'phone', speaker, save_file_name),
                            phone_index_list)
                    np.save(mkdir_join(save_path, 'phone_divide', speaker, save_file_name),
                            phone_divide_index_list)

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
