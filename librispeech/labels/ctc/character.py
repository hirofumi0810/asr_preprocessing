#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level target labels for the CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import re
import numpy as np
from tqdm import tqdm

from utils.labels.character import char2num
from utils.util import mkdir_join

# NOTE:
# - character
# 26 alphabets(a-z),
# space(_), apostorophe(')
# = 26 + 2 = 28 labels

# - character_capital_divide
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe(')
# = 26 * 2 + 24 + 1 = 77 labels


def read_text(label_paths, run_root_path, save_map_file=False, save_path=None,
              divide_by_capital=False):
    """Read text transcript.
    Args:
        label_paths: list of paths to label files
        run_root_path: absolute path of make.sh
        save_map_file: if True, save the mapping file
        save_path: path to save labels. If None, don't save labels
        divide_by_capital: if True, each word will be diveded by capital
            letters rather than spaces. In addition, repeated letters
            will be grouped by a special double-letter unit.
                ex.) hello => h e ll o
            This implementation is based on
                https://arxiv.org/abs/1609.05935.
                    Zweig, Geoffrey, et al.
                    "Advances in all-neural speech recognition."
                    in Proceedings of ICASSP, 2017.
    """
    print('===> Reading target labels...')
    speaker_dict = {}
    char_set = set([])
    for label_path in tqdm(label_paths):
        speaker_index = label_path.split('/')[-3]
        if speaker_index not in speaker_dict.keys():
            speaker_dict[speaker_index] = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                utterance_name = line[0]
                transcript = ' '.join(line[1:])

                if divide_by_capital:
                    transcript_capital = ''
                    for word in transcript.split(' '):
                        # Replace space with a capital letter
                        word = word[0].upper() + word[1:]

                        # Check double-letters
                        for i in range(0, len(word) - 1, 1):
                            if word[i] == word[i + 1]:
                                char_set.add(word[i] * 2)

                        transcript_capital += word
                    transcript = transcript_capital
                else:
                    # Convert space to "_"
                    transcript = re.sub(r'\s', '_', transcript)

                for c in list(transcript):
                    char_set.add(c)

                speaker_dict[speaker_index][utterance_name] = transcript

    # Make mapping file (from character to number)
    if divide_by_capital:
        mapping_file_path = join(
            run_root_path, 'labels/ctc/character2num_capital.txt')
    else:
        mapping_file_path = join(
            run_root_path, 'labels/ctc/character2num.txt')
    char_set.discard('_')
    char_set.discard('\'')

    # for debug
    print(sorted(list(char_set)))

    if save_map_file:
        with open(mapping_file_path, 'w') as f:
            if divide_by_capital:
                char_list = sorted(list(char_set)) + ['\'']
            else:
                char_list = ['_'] + sorted(list(char_set)) + ['\'']
            for index, char in enumerate(char_list):
                f.write('%s  %s\n' % (char, str(index)))

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for speaker_index, utterance_dict in tqdm(speaker_dict.items()):
            for utterance_name, transcript in utterance_dict.items():
                save_file_name = utterance_name + '.npy'

                # Convert from character to number
                char_index_list = char2num(transcript, mapping_file_path,
                                           double_letter=divide_by_capital)

                # Save as npy file
                mkdir_join(save_path, speaker_index)
                np.save(join(save_path, speaker_index, save_file_name),
                        char_index_list)
