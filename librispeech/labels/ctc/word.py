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
# -- train_clean100
# Original: 33798 labels
# Restricted: 7213 labels
# OOV rate: 6.85%
# -- train_clean360
# Original: 59661 labels
# Restricted: 16287 labels
# OOV rate: 3.16%
# -- train_other500
# Original: 66343 labels
# Restricted: 18669 labels
# OOV rate: 2.56%
# -- train_all
# Original: 89114 labels
# Restricted: 26642 labels
# OOV rate: 1.67%


def read_text(label_paths, data_type, train_data_size, run_root_path,
              is_test=False, save_map_file=False, save_path=None,
              frequency_threshold=5):
    """Read text transcript.
    Args:
        label_paths: list of paths to label files
        data_type: train_clean100 or train_clean360 or train_other500 or
            train_all or dev_clean or dev_other or test_clean or test_clean
        run_root_path: absolute path of make.sh
        is_test: bool, if False, restrict the vocaburary
        save_map_file: if True, save the mapping file
        save_path: path to save labels. If None, don't save labels
        frequency_threshold: int, the vocaburary is restricted to words which
            appear more than 'frequency_threshold' in the training set
    """
    print('===> Reading target labels...')
    speaker_dict = {}
    word_count_dict = {}
    word_set = set([])
    for label_path in tqdm(label_paths):
        speaker_index = label_path.split('/')[-3]
        if speaker_index not in speaker_dict.keys():
            speaker_dict[speaker_index] = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                uttrance_name = line[0]
                word_list = line[1:]

                for word in word_list:
                    word_set.add(word)
                    if word not in word_count_dict.keys():
                        word_count_dict[word] = 0
                    word_count_dict[word] += 1

                speaker_dict[speaker_index][uttrance_name] = word_list

    # Restrict the vocaburary
    if data_type in ['train_clean100', 'train_clean360',
                     'train_other500', 'train_all']:
        oov_list = [word for word, frequency in word_count_dict.items()
                    if frequency < frequency_threshold]
    elif data_type in ['dev_clean', 'dev_other']:
        oov_list = []
        with open(join(run_root_path, 'labels/ctc/word2num_' + train_data_size + '.txt'), 'r') as f:
            for line in f:
                line = line.strip().split()
                oov_list.append(line[0])
    else:
        # test
        oov_list = []

    # Make mapping file (from word to number)
    mapping_file_path = join(
        run_root_path, 'labels/ctc/word2num_' + train_data_size + '.txt')

    if save_map_file:
        with open(mapping_file_path, 'w') as f:
            restricted_word_set = word_set - set(oov_list)
            for index, word in enumerate(sorted(list(restricted_word_set)) + ['OOV']):
                f.write('%s  %s\n' % (word, str(index)))

        # for debug
        print('Original vocab: %d' % len(word_set))
        print('Restriced vocab: %d' % (len(restricted_word_set) + 1))
        total_word_count = np.sum(list(word_count_dict.values()))
        total_oov_word_count = np.sum(
            [count for word, count in word_count_dict.items() if word in oov_list])
        print('OOV rate %f %%' %
              ((total_oov_word_count / total_word_count) * 100))

    if save_path is not None:
        # Save target labels
        print('===> Saving target labels...')
        for speaker_index, utterance_dict in tqdm(speaker_dict.items()):
            for uttrance_name, word_list in utterance_dict.items():
                save_file_name = uttrance_name + '.npy'

                # Save as npy file
                mkdir_join(save_path, speaker_index)
                if is_test:
                    np.save(join(save_path, speaker_index, save_file_name),
                            word_list)
                    # NOTE: save a transcript as the list of words
                else:
                    # Convert to OOV
                    for i, word in enumerate(word_list):
                        if word in oov_list:
                            word_list[i] = 'OOV'

                    # Convert from word to number
                    word_index_list = word2num(word_list, mapping_file_path)

                    np.save(join(save_path, speaker_index, save_file_name),
                            word_index_list)


def convert_to_oov(word):
    global oov_list
