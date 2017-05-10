#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level labels for CTC model (eval2000 corpus)."""

import os
import sys
import re
import numpy as np
from tqdm import tqdm

import prepare_path
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
                value => [start_time, end_time, transcript]
    """
    print('===> Reading target labels...')
    speaker_dict = {}
    char_set = set([])
    for label_path in tqdm(label_paths):
        utterance_dict = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()

                # skip comment
                print(line)
                if line == '' or line[0] in ['#']:
                    continue

                line = line.split(' ')
                # utt_index = line[0].split('-')[-1]
                start_time = float(line[0])
                end_time = float(line[1])
                speaker_index = line[2]

                # convert to lowercase
                original_transcript = ' '.join(line[3:]).lower()

                print(original_transcript)

                # clean transcript
                # transcript = fix_transcript(original_transcript, speaker_index)

                # skip silence
                # if transcript == '':
                #     continue

                # merge silence around each utterance
                # transcript = '_' + transcript + '_'

                # for char in list(transcript):
                #     char_set.add(char)

                # utterance_dict[utt_index] = [start_time, end_time, transcript]
            # speaker_dict[speaker_index] = utterance_dict

    sys.exit(0)
    # make mapping file (from character to number)
    mapping_file_path = os.path.abspath('../mapping_files/char2num.txt')
    with open(mapping_file_path, 'w') as f:
        for index, char in enumerate(sorted(list(char_set))):
            f.write('%s  %s\n' % (char, str(index)))

    if save_path is not None:
        print('===> Saving target labels...')
        for speaker_index, utterance_dict in tqdm(speaker_dict.items()):
            save_path_speaker = mkdir(os.path.join(save_path, speaker_index))
            for utt_index,  utt_info in utterance_dict.items():
                start_time, end_time, transcript = utt_info
                save_file_name = speaker_index + '_' + utt_index + '.npy'

                # convert from character to number
                char_index_list = char2num(transcript, mapping_file_path)

                # save as npy file
                np.save(os.path.join(save_path_speaker, save_file_name), char_index_list)

    return speaker_dict


def fix_transcript(transcript, speaker_index):

    # remove <b_aside>, <e_aside>
    transcript = re.sub(r'\<b_aside\>', '', transcript)
    transcript = re.sub(r'\<e_aside\>', '', transcript)

    # replace [vocalized-noise], [noise], [silence] to the corresponding characters
    transcript = re.sub(r'\[vocalized-noise\]', 'VOCALIZED_NOISE', transcript)
    transcript = re.sub(r'\[noise\]', 'NOISE', transcript)
    transcript = re.sub(r'\[silence\]', '', transcript)

    ####################
    # laughter
    ####################
    # e.g. [LAUGHTER] -> 笑
    transcript = re.sub(r'\[laughter\]', 'LAUGHTER', transcript)

    # e.g. [LAUGHTER-STORY] -> 笑 STORY
    laughter_expr = re.compile(r'(.*)\[laughter-([\S]+)\](.*)')
    while re.match(laughter_expr, transcript) is not None:
        laughter = re.match(laughter_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = laughter.group(1) + 'LAUGHTER ' + laughter.group(2) + laughter.group(3)
        # print(transcript)
        # print('---')

    # exception (sw3845A)
    transcript = re.sub(r' laughter', ' LAUGHTER', transcript)

    ####################
    # which
    ####################
    # forward word is adopted
    # e.g. [IT'N/ISN'T] -> IT'N ... note,
    # 1st part may include partial-word stuff, which we process further below,
    # e.g. [LEM[GUINI]-/LINGUINI] -> LEM[GUINI]-
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
    # e.g. -[AN]Y
    backward_expr = re.compile(r'(.*)-\[([\S]+)\](.*)')
    while re.match(backward_expr, transcript) is not None:
        backward = re.match(backward_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = backward.group(1) + '-' + backward.group(3)
        # print(transcript)
        # print('---')

    # e.g. AB[SOLUTE]- -> AB-
    # e.g. EX[SPECIALLY]- -> EX-
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
    # e.g. {YUPPIEDOM} -> YUPPIEDOM
    nami_kakko_expr = re.compile(r'(.*)\{([\S]+)\}(.*)')
    while re.match(nami_kakko_expr, transcript) is not None:
        nami_kakko = re.match(nami_kakko_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = nami_kakko.group(1) + nami_kakko.group(2) + nami_kakko.group(3)
        # print(transcript)
        # print('---')

    # e.g. AMMU[N]IT- -> AMMU-IT- (sw2434A)
    kaku_kakko_expr = re.compile(r'(.*)\[([\S]+)\](.*)')
    while re.match(kaku_kakko_expr, transcript) is not None:
        kaku_kakko = re.match(kaku_kakko_expr, transcript)
        # print(speaker_index)
        # print(transcript)
        transcript = kaku_kakko.group(1) + '-' + kaku_kakko.group(3)
        # print(transcript)
        # print('---')

    # e.g. THEM_1 -> THEM
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


def test():
    prep = prepare_path.Prepare()
    trans = prep.label_train(label_type='character')
    # words = prep.label_train(label_type='word')

    read_trans(label_paths=trans)


if __name__ == '__main__':
    test()
