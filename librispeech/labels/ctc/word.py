#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make word-level target labels for the CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import numpy as np
from tqdm import tqdm

from utils.labels.word import word2num

# NOTE:
# ?? labels


def read_text(label_paths, data_type, run_root_path, save_map_file=False,
              save_path=None):
    """Read text transcript.
    Args:
        label_paths: list of paths to label files
        data_type: train_clean100 or train_clean360 or train_other500
            or dev_clean or dev_other or test_clean or test_clean
        run_root_path: absolute path of make.sh
        save_map_file: if True, save the mapping file
        save_path: path to save labels. If None, don't save labels
    """
    print('===> Reading target labels...')
    utterance_dict = {}
    word_set = set([])
    char_set = set([])
    for label_path in tqdm(label_paths):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                speaker_name = line[0]
                word_list = line[1:]

                for word in word_list:
                    word_set.add(word)
                    for c in word:
                        char_set.add(c)

                utterance_dict[speaker_name] = word_list

    # Make mapping file (from word to number)
    mapping_file_path = join(
        run_root_path, 'labels/ctc/word2num' + data_type + '.txt')

    if save_map_file:
        with open(mapping_file_path, 'w') as f:
            for index, word in enumerate(sorted(list(word_set))):
                f.write('%s  %s\n' % (word, str(index)))

    # for debug
    print(len(word_set))
    print(sorted(list(char_set)))

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for speaker_name, transcript in tqdm(utterance_dict.items()):
            save_file_name = speaker_name + '.npy'

            # Convert from word to number
            word_index_list = word2num(transcript, mapping_file_path)

            # Save as npy file
            np.save(join(save_path, save_file_name), word_index_list)
