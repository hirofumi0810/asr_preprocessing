#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Make phone-level labels for CTC model (LDC97S62 corpus)."""

import os
import re
import numpy as np
from tqdm import tqdm

from prepare_path import Prepare
from utils.util import mkdir
from utils.labels.phone import phone2num


# NOTE:
# 42 phones,
# SIL, LAUGHTER, NOISE
# = 42 + 3 = 45 labels


def read_pronounce_dict(dict_path, mapping_file_path):
    """Read pronounce dictionary.
    Args:
        dict_path: path to a pronounce dictionary
        mapping_file_path: path to save a mapping file
    Returns:
        pronounce_dict: pronounce dictionary
    """
    pronounce_dict = {}

    with open(dict_path, 'r') as f:
        for line in f:
            line = line.strip()

            # remove comment
            if line == '' or line[0] in ['#', ' ']:
                continue

            line = line.split(' ')
            word = line[0]
            phone_seq = ' '.join(line[1:])

            # remove head space
            if phone_seq[0] == ' ':
                phone_seq = phone_seq[1:]

            # add laughter label
            laughter_expr = re.compile(r'(.*)(\[laughter-[\S]+\])(.*)')
            if re.match(laughter_expr, word) is not None:
                phone_seq = 'LAUGHTER ' + phone_seq

            pronounce_dict[word.lower()] = phone_seq

    # add [vocalized-noise], [noise], [laughter] to the pronounce dictionary
    # pronounce_dict['[vocalized-noise]'] = 'VOCALIZED_NOISE'
    pronounce_dict['[vocalized-noise]'] = 'NOISE'
    pronounce_dict['[noise]'] = 'NOISE'
    pronounce_dict['[laughter]'] = 'LAUGHTER'

    # add single letters
    with open(mapping_file_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            letter = line[0]
            phone_seq = ' '.join(line[1:])
            pronounce_dict[letter.lower()] = phone_seq

    return pronounce_dict


def read_trans(label_paths, save_path=None):
    """Read transcripts (*_trans.txt) & save files (.npy).
    Args:
        label_paths: list of paths to label files
        save_path: path to save labels. If None, don't save labels
    Returns:
        speaker_dict: dictionary of speakers
            key => speaker name
            value => dictionary of utterance infomation of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
    """
    # read pronounce_dict
    print('===> Reading pronounce dictionary...')
    prep = Prepare()
    mapping_file_path = os.path.join(prep.run_root_path,
                                     'labels/MSU_single_letter.txt')
    pronounce_dict = read_pronounce_dict(
        prep.pronounce_dict_path, mapping_file_path)

    print('===> Reading target labels...')
    speaker_dict = {}
    phone_set = set([])
    for label_path in tqdm(label_paths):
        utterance_dict = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                speaker = line[0].split('-')[0]
                # Fix speaker name
                speaker = speaker.replace('sw0', 'sw').replace('A'. '-A').replace('-B', 'B')
                utt_index = line[0].split('-')[-1]
                start_frame = int(float(line[1]) * 100 + 0.05)
                end_frame = int(float(line[2]) * 100 + 0.05)

                # convert to lowercase
                original_transcript = ' '.join(line[3:]).lower()

                # clean transcript
                transcript = fix_transcript(original_transcript, speaker)

                # skip silence
                if transcript == '':
                    continue

                # remove head & last space
                if transcript[0] == ' ':
                    transcript = transcript[1:]
                if transcript[-1] == ' ':
                    transcript = transcript[:-1]

                # convert from character to phone
                phone_list = ['SIL']
                keys = pronounce_dict.keys()
                for word in transcript.split(' '):
                    if word in keys:
                        phone_list.append(pronounce_dict[word])
                        phone_list.append('SIL')
                    else:
                        print(transcript.split(' '))

                # convert to phone list where each element is phone(remove ' ')
                phone_seq = ' '.join(phone_list)
                phone_list = phone_seq.split(' ')

                for phone in phone_list:
                    if phone == '':
                        print(phone_list)
                    phone_set.add(phone)

                utterance_dict[utt_index.zfill(4)] = [
                    start_frame, end_frame, phone_list]
            speaker_dict[speaker] = utterance_dict

    return speaker_dict
