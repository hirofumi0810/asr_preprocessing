#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make labels for CTC model (monolog)."""

import os
import re
import codecs
import numpy as np
from tqdm import tqdm

from prepare_path import Prepare
from utils.util import mkdir
from utils.labels.character import kana2num
from utils.labels.phone import phone2num
from .fix_trans import fix_transcript


# NOTE:
# [character]
# 145 kana characters, space(_),
# = 145 + 1 + = 146 labels

# [phone]
#
#

def read_sdb(label_paths, label_type, save_path=None):
    """Read transcripts (.sdb) & save as npy files.
    Args:
        label_paths: list of paths to label files
        label_type: character or phone
        save_path: path to save labels. If None, don't save labels
    Returns:
        speaker_dict: dictionary of speakers
            key => speaker name
            value => dictionary of utterance infomation of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
    """
    print('===> Reading target labels...')
    speaker_dict = {}
    char_set = set([])
    for i, label_path in enumerate(tqdm(label_paths)):
        utterance_dict = {}
        utt_index_pre = 1
        start_frame_pre = None
        transcript = ''
        with codecs.open(label_path, 'r', 'shift-jis') as f:
            # print(label_path)
            for line in f:
                speaker_name = os.path.basename(label_path).split('.')[0]
                line = line.strip().split('\t')
                time_info = line[3].split(' ')
                utt_index = int(time_info[0])
                segment = time_info[1].split('-')
                start_frame = int(float(segment[0]) * 100)
                end_frame = int(float(segment[1]) * 100)
                if start_frame_pre is None:
                    start_frame_pre = start_frame

                # word = line[5]  # include kanji characters
                yomi = line[10]
                # pos_tag = line[11] if len(line) >= 12 else None

                # stack word in the same utterance
                if utt_index == utt_index_pre:
                    transcript += yomi
                    utt_index_pre = utt_index
                    continue
                else:
                    # count the number of kakko
                    left = transcript.count('(')
                    right = transcript.count(')')

                    if left != right:
                        transcript += yomi
                        utt_index_pre = utt_index
                        continue
                    else:
                        # clean transcript
                        transcript_fixed = fix_transcript(transcript)

                        # skip silence
                        if transcript_fixed != '':
                            # merge silence around each utterance
                            transcript_fixed = '_' + transcript_fixed + '_'

                            # remove double underbar
                            transcript_fixed = re.sub(
                                '__', '_', transcript_fixed)

                            for char in list(transcript_fixed):
                                char_set.add(char)

                            utterance_dict[utt_index] = [
                                start_frame_pre, end_frame, transcript_fixed]

                        # initialization
                        transcript = yomi
                        utt_index_pre = utt_index
                        start_frame_pre = None

            # register all utterances of each speaker
            speaker_dict[speaker_name] = utterance_dict

    # make mapping dictionary from kana to phone
    prep = Prepare()
    kana2phone_mapping_file_path = os.path.join(prep.run_root_path,
                                                'kana2phone.txt')
    kana_list = ['_']
    kana2phone_dict = {}
    phone_set = set([])
    with open(kana2phone_mapping_file_path, 'r') as f:
        for line in f:
            line = line.strip().split('+')
            kana, phone_seq = line
            kana_list.append(kana)
            kana2phone_dict[kana] = phone_seq
            for phone in phone_seq.split(' '):
                phone_set.add(phone)
        kana2phone_dict['_'] = '_'

    # make the mapping file (from kana character(phone) to number)
    if label_type == 'character':
        file_name = 'char2num.txt'
    elif label_type == 'phone':
        file_name = 'phone2num.txt'
    mapping_file_path = os.path.join(
        prep.run_root_path, 'labels/ctc', file_name)
    with open(mapping_file_path, 'w') as f:
        if label_type == 'character':
            for index, char in enumerate(kana_list):
                f.write('%s  %s\n' % (char, str(index)))
        elif label_type == 'phone':
            for index, phone in enumerate(sorted(list(phone_set))):
                if index == 0:
                    f.write('_  0\n')
                f.write('%s  %s\n' % (phone, str(index + 1)))

    # for debug
    # for char in list(char_set):
    #     if char not in kana_list:
    #         print(char)

    if save_path is not None:
        # save target labels
        print('===> Saving target labels...')
        for speaker_name, utterance_dict in tqdm(speaker_dict.items()):
            mkdir(os.path.join(save_path, speaker_name))
            for utt_index, utt_info in utterance_dict.items():
                start_frame, end_frame, transcript = utt_info
                save_file_name = speaker_name + '_' + str(utt_index) + '.npy'

                if label_type == 'character':
                    # convert from kana character to number
                    index_list = kana2num(transcript, mapping_file_path)

                elif label_type == 'phone':
                    # convert kana character to phone
                    trans_kana_list = list(transcript)
                    trans_phone_seq_list = []
                    i = 0
                    while i < len(trans_kana_list):
                        # check whether next character is a double consonant
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
                                    'There are no character such as %s' % trans_kana_list[i])
                        else:
                            if trans_kana_list[i] in kana2phone_dict.keys():
                                trans_phone_seq_list.append(
                                    kana2phone_dict[trans_kana_list[i]])
                            else:
                                raise ValueError(
                                    'There are no character such as %s' % trans_kana_list[i])
                        i += 1
                    trans_phone_list = []
                    for phone_seq in trans_phone_seq_list:
                        trans_phone_list.extend(phone_seq.split(' '))

                    # convert from phone to number
                    index_list = phone2num(trans_phone_list, mapping_file_path)

                # save as npy file
                np.save(os.path.join(save_path, speaker_name,
                                     save_file_name), index_list)

    return speaker_dict
