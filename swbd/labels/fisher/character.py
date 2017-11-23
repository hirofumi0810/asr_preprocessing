#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels for End-to-End model (Fisher corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import re
from tqdm import tqdm
from collections import OrderedDict

from swbd.labels.fisher.fix_trans import fix_transcript


DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
HYPHEN = '-'
APOSTROPHE = '\''
LAUGHTER = 'LA'
NOISE = 'NZ'
VOCALIZED_NOISE = 'VN'
OOV = 'OOV'


def read_trans(label_paths, target_speaker):
    """Read transcripts (*_trans.txt) & save files (.npy).
    Args:
        label_paths: list of paths to label files
        target_speaker: A or B
    Returns:
        speaker_dict: dictionary of speakers
            key (string) => speaker
            value (dict) => dictionary of utterance infomation of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
        char_set (set):
        char_capital_set (set):
        word_count_dict (dict):
            key => word
            value => the number of words in Fisher corpus
    """
    print('=====> Processing target labels...')
    speaker_dict = OrderedDict()
    char_set, char_capital_set = set([]), set([])
    word_count_dict = {}
    vocab_set = set([])

    for label_path in tqdm(label_paths):
        utterance_dict = OrderedDict()
        with open(label_path, 'r') as f:
            utt_index = 0
            session = basename(label_path).split('.')[0]
            for line in f:
                line = line.strip().split(' ')
                if line[0] in ['#', '']:
                    continue
                start_frame = int(float(line[0]) * 100 + 0.05)
                end_frame = int(float(line[1]) * 100 + 0.05)
                which_speaker = line[2].replace(':', '')
                if which_speaker != target_speaker:
                    continue
                speaker = session + which_speaker

                # Clean transcript
                transcript_original = ' '.join(line[3:]).lower()
                transcript = fix_transcript(transcript_original)

                # Skip silence
                if transcript in ['', ' ']:
                    continue

                # Skip laughter, noise, vocalized-noise only utterance
                if transcript.replace(NOISE, '').replace(SPACE, '').replace(VOCALIZED_NOISE, '') != '':

                    # Remove the first and last space
                    if transcript[0] == ' ':
                        transcript = transcript[1:]
                    if transcript[-1] == ' ':
                        transcript = transcript[:-1]

                    # Count words
                    for word in transcript.split(' '):
                        vocab_set.add(word)
                        if word not in word_count_dict.keys():
                            word_count_dict[word] = 0
                        word_count_dict[word] += 1

                    # Capital-divided
                    transcript_capital = ''
                    for word in transcript.split(' '):
                        if len(word) == 1:
                            char_capital_set.add(word)
                            transcript_capital += word
                        else:
                            # Replace the first character with the capital
                            # letter
                            word = word[0].upper() + word[1:]

                            # Check double-letters
                            for i in range(0, len(word) - 1, 1):
                                if word[i:i + 2] in DOUBLE_LETTERS:
                                    char_capital_set.add(word[i:i + 2])
                                else:
                                    char_capital_set.add(word[i])
                            transcript_capital += word

                    # Convert space to "_"
                    transcript = re.sub(r'\s', SPACE, transcript)

                    for c in list(transcript):
                        char_set.add(c)

                    utterance_dict[str(utt_index).zfill(4)] = [
                        start_frame, end_frame, transcript]

                    # for debug
                    # print(transcript_original)
                    # print(transcript)

                utt_index += 1

            speaker_dict[speaker] = utterance_dict

    # Reserve some indices
    for mark in [SPACE, HYPHEN, APOSTROPHE, LAUGHTER, NOISE, VOCALIZED_NOISE]:
        for c in list(mark):
            char_set.discard(c)
            char_capital_set.discard(c)

    # for debug
    # print(sorted(list(char_set)))
    # print(sorted(list(char_capital_set)))

    return speaker_dict, char_set, char_capital_set, word_count_dict
