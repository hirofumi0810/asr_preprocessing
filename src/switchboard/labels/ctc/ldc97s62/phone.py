#! /usr/bin/env python
# -*- coding: utf-8 -*-

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
# SILENCE, LAUGHTER, VOCALIZED_NOISE, NOISE
# = 42 + 4 = 46 labels


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
    pronounce_dict['[vocalized-noise]'] = 'VOCALIZED_NOISE'
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


def read_transcript(label_paths, save_path=None):
    """Read transcripts & save as npy files.
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
                speaker_name = line[0].split('-')[0]
                utt_index = line[0].split('-')[-1]
                start_frame = int(float(line[1]) * 100 + 0.05)
                end_frame = int(float(line[2]) * 100 + 0.05)

                # convert to lowercase
                original_transcript = ' '.join(line[3:]).lower()

                # clean transcript
                transcript = fix_transcript(original_transcript, speaker_name)

                # skip silence
                if transcript == '':
                    continue

                # remove head & last space
                if transcript[0] == ' ':
                    transcript = transcript[1:]
                if transcript[-1] == ' ':
                    transcript = transcript[:-1]

                # convert from character to phone
                phone_list = ['SILENCE']
                keys = pronounce_dict.keys()
                for word in transcript.split(' '):
                    if word in keys:
                        phone_list.append(pronounce_dict[word])
                        phone_list.append('SILENCE')
                    else:
                        print(transcript.split(' '))

                # convert to phone list where each element is phone (remove '
                # ')
                phone_seq = ' '.join(phone_list)
                phone_list = phone_seq.split(' ')

                for phone in phone_list:
                    if phone == '':
                        print(phone_list)
                    phone_set.add(phone)

                utterance_dict[utt_index] = [
                    start_frame, end_frame, phone_list]
            speaker_dict[speaker_name] = utterance_dict

    # make the mapping file (from phone to number)
    mapping_file_path = os.path.join(prep.run_root_path,
                                     'labels/ctc/phone2num.txt')
    with open(mapping_file_path, 'w') as f:
        for index, phone in enumerate(sorted(list(phone_set))):
            f.write('%s  %s\n' % (phone, str(index)))

    if save_path is not None:
        # save target labels
        print('===> Saving target labels...')
        for speaker_name, utterance_dict in tqdm(speaker_dict.items()):
            mkdir(os.path.join(save_path, speaker_name))
            for utt_index, utt_info in utterance_dict.items():
                start_frame, end_frame, phone_list = utt_info
                save_file_name = speaker_name + '_' + utt_index + '.npy'

                # convert from phone to number
                phone_index_list = phone2num(phone_list, mapping_file_path)

                # save as npy file
                np.save(os.path.join(save_path, speaker_name,
                                     save_file_name), phone_index_list)

    return speaker_dict


def fix_transcript(transcript, speaker_name):
    # remove <b_aside>, <e_aside>, [silence]
    transcript = re.sub(r'\<b_aside\>', '', transcript)
    transcript = re.sub(r'\<e_aside\>', '', transcript)
    transcript = re.sub(r'\[silence\]', '', transcript)

    # exception (sw3845A)
    if ' laughter' in transcript:
        transcript = re.sub(r' laughter', ' [laughter]', transcript)

    ####################
    # exception
    ####################
    # remove double space
    transcript = re.sub('  ', ' ', transcript)

    return transcript
