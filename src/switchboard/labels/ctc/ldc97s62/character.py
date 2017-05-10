#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level labels for CTC model (LDC97S62 corpus)."""

import os
import re
import numpy as np
from tqdm import tqdm

from prepare_path import Prepare
from utils.util import mkdir
from utils.labels.character import char2num


# NOTE:
# 26 alphabets(a-z), 10 numbers(0-9),
# space(_), apostorophe('), hyphen(-),
# L:laughter, V:vocalized-noise, N:noise
# = 26 + 10 + 3 + 3 = 42 labels


def read_transcript(label_paths, save_path=None):
    """Read transcripts & save as npy files.
    Args:
        label_paths: list of paths to label files
        save_path: path to save labels. If None, don't save labels
    Returns:
        speaker_dict: dictionary of speakers
            key => speaker index
            value => dictionary of utterance infomation of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
    """
    print('===> Reading target labels...')
    speaker_dict = {}
    char_set = set([])
    for label_path in tqdm(label_paths):
        utterance_dict = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                speaker_index = line[0].split('-')[0]
                utt_index = line[0].split('-')[-1]
                start_frame = int(float(line[1]) * 100 + 0.05)
                end_frame = int(float(line[2]) * 100 + 0.05)

                # convert to lowercase
                original_transcript = ' '.join(line[3:]).lower()

                # clean transcript
                transcript = fix_transcript(original_transcript, speaker_index)

                # skip silence
                if transcript == '':
                    continue

                # merge silence around each utterance
                transcript = '_' + transcript + '_'

                # remove double underbar
                transcript = re.sub('__', '_', transcript)

                for char in list(transcript.lower()):
                    char_set.add(char)

                utterance_dict[utt_index] = [start_frame, end_frame, transcript]
            speaker_dict[speaker_index] = utterance_dict

    # make the mapping file (from character to number)
    prep = Prepare()
    mapping_file_path = os.path.join(prep.run_root_path,
                                     'labels/ctc/char2num.txt')
    char_set.add('L')
    char_set.add('V')
    char_set.add('N')
    with open(mapping_file_path, 'w') as f:
        for index, char in enumerate(sorted(list(char_set))):
            f.write('%s  %s\n' % (char, str(index)))

    if save_path is not None:
        # save target labels
        print('===> Saving target labels...')
        for speaker_index, utterance_dict in tqdm(speaker_dict.items()):
            mkdir(os.path.join(save_path, speaker_index))
            for utt_index, utt_info in utterance_dict.items():
                start_frame, end_frame, transcript = utt_info
                save_file_name = speaker_index + '_' + utt_index + '.npy'

                # convert from character to number
                char_index_list = char2num(transcript, mapping_file_path)

                # save as npy file
                np.save(os.path.join(save_path, speaker_index, save_file_name), char_index_list)

    return speaker_dict


def fix_transcript(transcript, speaker_index):
    # remove <b_aside>, <e_aside>, [silence]
    transcript = re.sub(r'\<b_aside\>', '', transcript)
    transcript = re.sub(r'\<e_aside\>', '', transcript)
    transcript = re.sub(r'\[silence\]', '', transcript)

    # replace [vocalized-noise], [noise] to the corresponding characters
    transcript = re.sub(r'\[vocalized-noise\]', 'V', transcript)
    transcript = re.sub(r'\[noise\]', 'N', transcript)

    ####################
    # laughter
    ####################
    # e.g. [laughter] -> L
    transcript = re.sub(r'\[laughter\]', 'L', transcript)

    # e.g. [laughter-story] -> L story
    laughter_expr = re.compile(r'(.*)\[laughter-([\S]+)\](.*)')
    while re.match(laughter_expr, transcript) is not None:
        laughter = re.match(laughter_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = laughter.group(1) + 'L ' + laughter.group(2) + laughter.group(3)
        # print(transcript)
        # print('---')

    # exception (sw3845A)
    transcript = re.sub(r' laughter', ' L', transcript)

    ####################
    # which
    ####################
    # forward word is adopted
    # e.g. [it'n/isn't] -> it'n ... note,
    # 1st part may include partial-word stuff, which we process further below,
    # e.g. [lem[guini]-/linguini] -> lem[guini]-
    which_expr = re.compile(r'(.*)\[([\S]+)/([\S]+)\](.*)')
    while re.match(which_expr, transcript) is not None:
        which = re.match(which_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = which.group(1) + which.group(2) + which.group(4)
        # print(transcript)
        # print('---')

    ####################
    # disfluency
    # -で言い淀みを表現
    ####################
    # e.g. -[an]y
    backward_expr = re.compile(r'(.*)-\[([\S]+)\](.*)')
    while re.match(backward_expr, transcript) is not None:
        backward = re.match(backward_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = backward.group(1) + '-' + backward.group(3)
        # print(transcript)
        # print('---')

    # e.g. ab[solute]- -> ab-
    # e.g. ex[specially]- -> ex-
    forward_expr = re.compile(r'(.*)\[([\S]+)\]-(.*)')
    while re.match(forward_expr, transcript) is not None:
        forward = re.match(forward_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = forward.group(1) + '-' + forward.group(3)
        # print(transcript)
        # print('---')

    ####################
    # exception
    ####################
    # e.g. {yuppiedom} -> yuppiedom
    nami_kakko_expr = re.compile(r'(.*)\{([\S]+)\}(.*)')
    while re.match(nami_kakko_expr, transcript) is not None:
        nami_kakko = re.match(nami_kakko_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = nami_kakko.group(1) + nami_kakko.group(2) + nami_kakko.group(3)
        # print(transcript)
        # print('---')

    # e.g. ammu[n]it- -> ammu-it- (sw2434A)
    kaku_kakko_expr = re.compile(r'(.*)\[([\S]+)\](.*)')
    while re.match(kaku_kakko_expr, transcript) is not None:
        kaku_kakko = re.match(kaku_kakko_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = kaku_kakko.group(1) + '-' + kaku_kakko.group(3)
        # print(transcript)
        # print('---')

    # e.g. them_1 -> them
    transcript = re.sub(r'_\d', 'them', transcript)

    # replace "&" to "and"
    transcript = re.sub('&', ' and ', transcript)

    # remove "/"
    transcript = re.sub('/', '', transcript)

    # remove double space
    transcript = re.sub('  ', ' ', transcript)

    # replace space( ) to "_"
    transcript = re.sub(' ', '_', transcript)

    return transcript
