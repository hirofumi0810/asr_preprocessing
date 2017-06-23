#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make labels for CTC model (CSJ corpus)."""

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
from utils.labels.character import kana2num
from utils.labels.phone import phone2num
from .fix_trans import fix_transcript
from .fix_trans import is_hiragana, is_katakana, is_kanji, is_alphabet

# NOTE:
# [character]
# 145 kana characters, noise(NZ), space(_),
# = 145 + 2 + = 147 labels

# [phone]
# 36 phones, noise(NZ), sil(_),
# = 36 + 2 = 38 labels

# [kanji]
# ?? kanji characters, ?? kana characters, ?? hiragana characters,
# noise(NZ), sil(_),
# = ?? + ?? + ?? + 2 = 3386 lables


def read_sdb(label_paths, label_type, run_root_path, is_test=None,
             save_map_file=False, save_path=None):
    """Read transcripts (.sdb) & save files (.npy).
    Args:
        label_paths: list of paths to label files
        label_type: character or phone or kanji
        run_root_path: absolute path of make.sh
        is_test: set to True when making the test set
        save_map_file: if True, save the mapping file
        save_path: path to save labels. If None, don't save labels
    Returns:
        speaker_dict: the dictionary of utterances of each speaker
            key => speaker name
            value => the dictionary of utterance information of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
    """
    print('===> Loading target labels...')
    speaker_dict = {}
    char_set = set([])
    all_char_set = set([])
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
        transcript, transcript_kanji = '', ''
        speaker_name = basename(label_path).split('.')[0]
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
                transcript += yomi
                transcript_kanji += kanji
                # transcript_kanji += kanji + '_'
                # TODO: どっちが良いか要検討
                utt_index_pre = utt_index
                end_frame_pre = end_frame
                continue
            else:
                # Count the number of kakko
                left_kanji = transcript_kanji.count('(')
                right_kanji = transcript_kanji.count(')')
                if left_kanji != right_kanji:
                    transcript += yomi
                    transcript_kanji += kanji
                    utt_index_pre = utt_index
                    end_frame_pre = end_frame
                    continue

                left = transcript.count('(')
                right = transcript.count(')')
                if left != right:
                    transcript += yomi
                    transcript_kanji += kanji
                    utt_index_pre = utt_index
                    end_frame_pre = end_frame
                    continue
                else:
                    # Clean transcript
                    transcript = '_' + fix_transcript(transcript,
                                                      speaker_name) + '_'
                    transcript_kanji = '_' + fix_transcript(transcript_kanji,
                                                            speaker_name) + '_'

                    # Remove double underbar
                    while '__' in transcript:
                        transcript = re.sub('__', '_', transcript)
                    while '__' in transcript_kanji:
                        transcript_kanji = re.sub('__', '_', transcript_kanji)

                    # Skip silence & noise only utterance
                    if transcript not in ['_', '_NZ_']:
                        for char in list(transcript):
                            char_set.add(char)
                        for char in list(transcript_kanji):
                            all_char_set.add(char)

                        if label_type == 'kanji':
                            utterance_dict[str(utt_index - 1).zfill(4)] = [
                                start_frame_pre, end_frame_pre, transcript_kanji]
                        else:
                            utterance_dict[str(utt_index - 1).zfill(4)] = [
                                start_frame_pre, end_frame_pre, transcript]

                    # Initialization
                    transcript = yomi
                    transcript_kanji = kanji
                    utt_index_pre = utt_index
                    start_frame_pre = start_frame
                    end_frame_pre = end_frame

        # Register all utterances of each speaker
        speaker_dict[speaker_name] = utterance_dict

    # Make mapping dictionary from kana to phone
    kana_list = []
    kana2phone_dict = {}
    phone_set = set([])
    with open(join(run_root_path, 'kana2phone.txt'), 'r') as f:
        for line in f:
            line = line.strip().split('+')
            kana, phone_seq = line
            kana_list.append(kana)
            kana2phone_dict[kana] = phone_seq
            for phone in phone_seq.split(' '):
                phone_set.add(phone)
        kana2phone_dict['_'] = '_'
        kana2phone_dict['NZ'] = 'NZ'

    # Make the mapping file (from kanji, kana, phone to number)
    if label_type == 'kanji':
        file_name = 'kanji2num.txt'
    elif label_type == 'character':
        file_name = 'char2num.txt'
    elif label_type == 'phone':
        file_name = 'phone2num.txt'
    mapping_file_path = join(run_root_path, 'labels/ctc', file_name)
    if save_map_file:
        with open(mapping_file_path, 'w') as f:
            if label_type == 'kanji':
                # インデックスを予約するラベル
                all_char_set.discard('N')
                all_char_set.discard('Z')
                all_char_set.discard('_')

                kanji_set = set([])
                for char in all_char_set:
                    if (not is_hiragana(char)) and (not is_katakana(char)):
                        kanji_set.add(char)
                for char in kana_list:
                    kanji_set.add(char)
                    kanji_set.add(jaconv.kata2hira(char))
                # NOTE: 頻出するラベルにはなるべく小さいインデックスを与える
                kanji_list = ['_', 'NZ'] + sorted(list(kanji_set))
                for index, kanji in enumerate(kanji_list):
                    f.write('%s  %s\n' % (kanji, str(index)))

            elif label_type == 'character':
                kana_list = ['_', 'NZ'] + kana_list
                for index, char in enumerate(kana_list):
                    f.write('%s  %s\n' % (char, str(index)))

            elif label_type == 'phone':
                phone_list = ['_', 'NZ'] + sorted(list(phone_set))
                for index, phone in enumerate(phone_list):
                    f.write('%s  %s\n' % (phone, str(index)))

    # for debug
    for char in list(char_set):
        if char not in kana_list:
            print(char)

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for speaker_name, utterance_dict in tqdm(speaker_dict.items()):
            mkdir_join(save_path, speaker_name)
            for utt_index, utt_info in utterance_dict.items():
                start_frame, end_frame, transcript = utt_info
                save_file_name = speaker_name + '_' + utt_index + '.npy'

                if label_type == 'kanji':
                    if not is_test:
                        # Convert from kana character to number
                        index_list = kana2num(transcript, mapping_file_path)

                        # Save as npy file
                        np.save(join(save_path, speaker_name,
                                     save_file_name), index_list)
                    else:
                        # NOTE: テストデータは文字列としてそのまま保存
                        # Save as npy file
                        np.save(join(save_path, speaker_name,
                                     save_file_name), transcript)

                elif label_type == 'character':
                    if not is_test:
                        # Convert from kana character to number
                        index_list = kana2num(transcript, mapping_file_path)

                        # Save as npy file
                        np.save(join(save_path, speaker_name,
                                     save_file_name), index_list)
                    else:
                        # NOTE: テストデータは文字列としてそのまま保存
                        # Save as npy file
                        np.save(join(save_path, speaker_name,
                                     save_file_name), transcript)

                elif label_type == 'phone':
                    # Convert kana character to phone
                    trans_kana_list = list(transcript)
                    trans_phone_seq_list = []
                    i = 0
                    while i < len(trans_kana_list):
                        # Check whether next character is a double consonant
                        if i != len(trans_kana_list) - 1:
                            if trans_kana_list[i] + trans_kana_list[i + 1] in kana2phone_dict.keys():
                                trans_phone_seq_list.append(
                                    kana2phone_dict[trans_kana_list[i] + trans_kana_list[i + 1]])
                                i += 1
                            elif trans_kana_list[i] in kana2phone_dict.keys():
                                trans_phone_seq_list.append(
                                    kana2phone_dict[trans_kana_list[i]])
                            else:
                                raise ValueError(
                                    'There are no character such as %s'
                                    % trans_kana_list[i])
                        else:
                            if trans_kana_list[i] in kana2phone_dict.keys():
                                trans_phone_seq_list.append(
                                    kana2phone_dict[trans_kana_list[i]])
                            else:
                                raise ValueError(
                                    'There are no character such as %s'
                                    % trans_kana_list[i])
                        i += 1
                    trans_phone_list = []
                    for phone_seq in trans_phone_seq_list:
                        trans_phone_list.extend(phone_seq.split(' '))

                    # Convert from phone to number
                    index_list = phone2num(trans_phone_list, mapping_file_path)

                    # Save as npy file
                    np.save(join(save_path, speaker_name,
                                 save_file_name), index_list)

    return speaker_dict
