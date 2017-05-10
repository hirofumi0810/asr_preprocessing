#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level labels for CTC model (eval2000 corpus)."""

import os
import re
import numpy as np
from tqdm import tqdm

from prepare_path import Prepare
from utils.util import mkdir
from utils.labels.character import char2num


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
        flag_speaker_b = False
        utt_index = 0
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                start_frame = int(float(line[0]) * 100)
                end_frame = int(float(line[1]) * 100)
                if line[2] == 'B:' and not flag_speaker_b:
                    flag_speaker_b = True
                    speaker_index_a = os.path.basename(label_path).split('.')[0] + 'A'
                    speaker_index_a = speaker_index_a.replace('sw_', 'sw')
                    speaker_dict[speaker_index_a] = utterance_dict
                    # reset
                    utterance_dict = {}
                    utt_index = 0
                speaker_index = os.path.basename(label_path).split('.')[0] + line[2][0]
                speaker_index = speaker_index.replace('sw_', 'sw')

                # convert to lowercase
                original_transcript = ' '.join(line[3:]).lower()

                # clean transcript
                transcript = fix_transcript(original_transcript, speaker_index)

                # merge silence around each utterance
                # transcript = '_' + transcript + '_'

                # remove double underbar
                transcript = re.sub('__', '_', transcript)

                for char in list(transcript.lower()):
                    char_set.add(char)

                utterance_dict[str(utt_index)] = [start_frame, end_frame, transcript]
                utt_index += 1
            speaker_dict[speaker_index] = utterance_dict

    # print(sorted(list(char_set)))

    # read mapping file (from character to number)
    prep = Prepare()
    mapping_file_path = os.path.join(prep.run_root_path,
                                     'labels/ctc/char2num.txt')

    if save_path is not None:
        # save target labels
        print('===> Saving target labels...')
        for speaker_index, utterance_dict in tqdm(speaker_dict.items()):
            save_path_speaker = mkdir(os.path.join(save_path, speaker_index))
            for utt_index,  utt_info in utterance_dict.items():
                start_frame, end_frame, transcript = utt_info
                save_file_name = speaker_index + '_' + utt_index + '.npy'

                # convert from character to number
                char_index_list = char2num(transcript, mapping_file_path)

                # save as npy file
                np.save(os.path.join(save_path_speaker, save_file_name), char_index_list)

    return speaker_dict


def fix_transcript(transcript, speaker_index):

    # remove <b_aside>, <e_aside>, [silence], [vocalized-noise], [noise], [noise-good]
    transcript = re.sub(r'\<b_aside\>', '', transcript)
    transcript = re.sub(r'\<e_aside\>', '', transcript)
    transcript = re.sub(r'\[silence\]', '', transcript)
    transcript = re.sub(r'\[vocalized-noise\]', '', transcript)
    transcript = re.sub(r'\[noise\]', '', transcript)
    transcript = re.sub(r'\[noise-good\]', '', transcript)

    # NOTE:とりあえず今は[uh]を消す
    transcript = re.sub(r'\[uh\]', '', transcript)

    ####################
    # laughter
    ####################
    # e.g. [LAUGHTER] -> L
    transcript = re.sub(r'\[laughter\]', 'L', transcript)

    # e.g. [LAUGHTER-STORY] -> L STORY
    laughter_expr = re.compile(r'(.*)\[laughter-([\S]+)\](.*)')
    while re.match(laughter_expr, transcript) is not None:
        laughter = re.match(laughter_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = laughter.group(1) + 'L ' + laughter.group(2) + laughter.group(3)
        # print(transcript)
        # print('---')

    # NOTE:とりあえず今は L を消す
    transcript = re.sub(r'L', '', transcript)

    ####################
    # abbreviation
    ####################
    # e.g. i'm -> i am (2 words)
    abbr_expr2 = re.compile(
        r'(.*)<contraction e_form=\"\[[\S]+=>([\S]+)\]\[[\S]+=>([\S]+)\]\">([\S]+)(.*)')
    while re.match(abbr_expr2, transcript) is not None:
        abbr = re.match(abbr_expr2, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = abbr.group(1) + abbr.group(2) + ' ' + abbr.group(3) + abbr.group(5)
        # print(transcript)
        # print('---')

    # e.g. can't -> cannot (1 word)
    abbr_expr1 = re.compile(
        r'(.*)<contraction e_form=\"\[[\S]+=>([\S]+)\]\">([\S]+)(.*)')
    while re.match(abbr_expr1, transcript) is not None:
        abbr = re.match(abbr_expr1, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = abbr.group(1) + abbr.group(2) + abbr.group(4)
        # print(transcript)
        # print('---')

    ####################
    # double bracket
    ####################
    # e.g. ((yeah)) -> yeah
    bracket_expr = re.compile(r'(.*)\(\((.+)\)\)(.*)')
    while re.match(bracket_expr, transcript) is not None:
        bracket = re.match(bracket_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = bracket.group(1) + bracket.group(2) + bracket.group(3)
        # print(transcript)
        # print('---')

    # e.g. ((is => is
    transcript = re.sub(r'\(\(', '', transcript)
    transcript = re.sub(r'\)', '', transcript)

    ####################
    # disfluency
    # -で言い淀みを表現
    ####################
    # e.g. [right] -> right
    transcript = re.sub(r'\[right\]', 'right', transcript)

    # -y[ou]i-
    middle_expr = re.compile(r'(.*) ([\S]+)\[([\S]+)\]([\S]+)- (.*)')
    while re.match(middle_expr, transcript) is not None:
        middle = re.match(middle_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = middle.group(1) + ' ' + middle.group(2) + \
            middle.group(4) + '- ' + middle.group(5)
        # print(transcript)
        # print('---1')

    # e.g. -[w]here -> -here
    # e.g. -[a]nd -> -nd
    backward_expr = re.compile(r'(.*)-\[([\S]+)\](.*)')
    while re.match(backward_expr, transcript) is not None:
        backward = re.match(backward_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = backward.group(1) + '-' + backward.group(3)
        # print(transcript)
        # print('---2')

    # e.g. ju[st] -> ju-
    # e.g. rein[carnating] -> rein-
    forward_expr = re.compile(r'(.*) ([\S]+)\[([\S]+)\] (.*)')
    while re.match(forward_expr, transcript) is not None:
        forward = re.match(forward_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = forward.group(1) + ' ' + forward.group(2) + '- ' + forward.group(4)
        # print(transcript)
        # print('---3')

    ####################
    # exception
    ####################

    # remove double space
    transcript = re.sub('  ', ' ', transcript)

    # replace space( ) to "_"
    transcript = re.sub(' ', '_', transcript)

    return transcript
