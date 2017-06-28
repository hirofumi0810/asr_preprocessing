#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make target labels for the CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import re
import numpy as np
from tqdm import tqdm

from utils.labels.character import char2num

# NOTE:
# 26 alphabets(a-z),
# space(_), comma(,), period(.), apostorophe('), hyphen(-),
# question(?), exclamation(!)
# = 26 + 7 = 33 labels


def read_text(label_paths, run_root_path, save_map_file=False, save_path=None):
    """Read text transcript.
    Args:
        label_paths: list of paths to label files
        run_root_path: absolute path of make.sh
        save_map_file: if True, save the mapping file
        save_path: path to save labels. If None, don't save labels
    """
    print('===> Reading target labels...')
    text_dict = {}
    char_set = set([])
    for label_path in tqdm(label_paths):
        with open(label_path, 'r') as f:
            line = f.readlines()[-1]

            # Remove 「"」, 「:」, 「;」
            # Convert to lowercase
            line = re.sub(r'[\":;]+', '', line.strip().lower())

            # Convert space to "_"
            transcript = '_' + '_'.join(line.split(' ')[2:]) + '_'

        for c in list(transcript):
            char_set.add(c)

        text_dict[label_path] = transcript

    # Make mapping file (from character to number)
    mapping_file_path = join(run_root_path, 'labels/ctc/char2num.txt')
    char_set.discard('_')
    char_set.discard(',')
    char_set.discard('.')
    char_set.discard('\'')
    char_set.discard('-')
    char_set.discard('?')
    char_set.discard('!')

    if save_map_file:
        with open(mapping_file_path, 'w') as f:
            char_list = ['_'] + sorted(list(char_set))
            char_list += [',', '.', '\'', '-', '?', '!']
            for index, char in enumerate(char_list):
                f.write('%s  %s\n' % (char, str(index)))

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for label_path, transcript in tqdm(text_dict.items()):
            speaker_name = label_path.split('/')[-2]
            file_name = label_path.split('/')[-1].split('.')[0]
            save_file_name = speaker_name + '_' + file_name + '.npy'

            # Convert from character to number
            char_index_list = char2num(transcript, mapping_file_path)

            # Save as npy file
            np.save(join(save_path, save_file_name), char_index_list)
