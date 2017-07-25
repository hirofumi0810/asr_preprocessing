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
from utils.util import mkdir_join

# NOTE:
# - train_clean100
# ?? labels
# - train_clean360
# ?? labels
# - train_other500
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
    speaker_dict = {}
    word_set = set([])
    char_set = set([])
    for label_path in tqdm(label_paths):
        speaker_index = label_path.split('/')[-3]
        if speaker_index in speaker_dict.keys():
            speaker_dict[speaker_index] = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                uttrance_name = line[0]
                word_list = line[1:]

                for word in word_list:
                    word_set.add(word)
                    for c in word:
                        char_set.add(c)

                speaker_dict[speaker_index][uttrance_name] = word_list

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
        for speaker_index, utterance_dict in tqdm(speaker_dict.items()):
            for uttrance_name, transcript in utterance_dict.items():
                save_file_name = uttrance_name + '.npy'

                # Convert from word to number
                word_index_list = word2num(transcript, mapping_file_path)

                # Save as npy file
                mkdir_join(save_path, speaker_index)
                np.save(join(save_path, speaker_index, save_file_name),
                        word_index_list)
