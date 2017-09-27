#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make word-level target labels for the End-to-End model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm

from utils.labels.word import Word2idx
from utils.util import mkdir_join

# NOTE:
############################################################
# CTC model

# -- threshold == 10

# [train_clean100]
# Original: 33798 labels
# Restricted: 7213 labels
# OOV rate: 6.85%

# [train_clean360]
# Original: 59661 labels
# Restricted: 16287 labels
# OOV rate: 3.16%

# [train_other500]
# Original: 66343 labels
# Restricted: 18669 labels
# OOV rate: 2.56%

# [train_all]
# Original: 89114 labels
# Restricted: 26642 labels
# OOV rate: 1.67%
############################################################

############################################################
# Attention-based model

# [train_clean100]
# Restricted: 7215 labels

# [train_clean360]
# Restricted: 16289 labels

# [train_other500]
# Restricted: 18671 labels

# [train_all]
# Restricted: 26644 labels
############################################################


def read_word(label_paths, data_type, train_data_size, run_root_path, model,
              save_map_file=False, save_path=None,
              frequency_threshold=5, stdout_transcript=False):
    """Read text transcript.
    Args:
        label_paths (list): list of paths to label files
        data_type (string): train_clean100 or train_clean360 or train_other500
            or train_all or dev_clean or dev_other or test_clean or test_clean
        run_root_path (string): absolute path of make.sh
        model (string): ctc or attention
        vocab_path: (string): path to the vocabulary dict
        save_map_file (bool ,optional): if True, save the mapping file
        save_path (string, optional): path to save labels. If None, don't save
            labels
        frequency_threshold (int, optional): the vocabulary is restricted to
            words which appear more than 'frequency_threshold' in the training
            set
        stdout_transcript (bool, optional): if True, print processed
            transcripts to standard output
    """
    if model not in ['ctc', 'attention']:
        raise ValueError('model must be ctc or attention.')

    is_training, is_test = False, False
    if data_type in ['train_clean100', 'train_clean360',
                     'train_other500', 'train_all']:
        is_training = True
    elif data_type in ['test_clean', 'test_other']:
        is_test = True

    print('===> Reading target labels...')
    speaker_dict = {}
    word_count_dict = {}
    vocab_set = set([])
    for label_path in tqdm(label_paths):
        speaker = label_path.split('/')[-3]
        if speaker not in speaker_dict.keys():
            speaker_dict[speaker] = {}
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                utt_name = line[0]  # ex.) speaker-book-utt_index
                word_list = line[1:]

                if is_training:
                    for word in word_list:
                        vocab_set.add(word)
                        if word not in word_count_dict.keys():
                            word_count_dict[word] = 0
                        word_count_dict[word] += 1

                if model == 'attention':
                    word_list = ['<'] + word_list + ['>']

                speaker_dict[speaker][utt_name] = word_list

                if stdout_transcript:
                    print(' '.join(word_list))

    map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', model,
        'word_' + train_data_size + '.txt')

    if is_training:
        # Restrict the vocabulary
        oov_list = [word for word, frequency in word_count_dict.items()
                    if frequency < frequency_threshold]
        original_vocab_num = len(vocab_set)
        vocab_set -= set(oov_list)
    else:
        # Read the mapping file
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                vocab_set.add(line[0])

    # Make mapping file (from word to index)
    if save_map_file and is_training:
        word_list = sorted(list(vocab_set)) + ['OOV']
        if model == 'attention':
            word_list = ['<', '>'] + word_list
        with open(map_file_path, 'w') as f:
            for i, word in enumerate(word_list):
                f.write('%s  %s\n' % (word, str(i)))

        # for debug
        print('Original vocab: %d' % original_vocab_num)
        print('Restriced vocab: %d' % (len(vocab_set) + 1))  # + OOV
        total_word_count = np.sum(list(word_count_dict.values()))
        total_oov_word_count = np.sum(
            [count for word, count in word_count_dict.items() if word in oov_list])
        print('OOV rate %f %%' %
              ((total_oov_word_count / total_word_count) * 100))

    if save_path is not None:
        word2idx = Word2idx(map_file_path=map_file_path)

        # Save target labels
        print('===> Saving target labels...')
        for speaker, utterance_dict in tqdm(speaker_dict.items()):
            for utt_name, word_list in utterance_dict.items():

                # Save as npy file
                if is_test:
                    np.save(mkdir_join(save_path, speaker, utt_name + '.npy'),
                            ' '.join(word_list))
                    # NOTE: save a transcript as string
                else:
                    # Convert to OOV
                    word_list = [
                        word if word in vocab_set else 'OOV' for word in word_list]

                    # Convert from word to index
                    word_index_list = word2idx(word_list)

                    np.save(mkdir_join(save_path, speaker, utt_name + '.npy'),
                            word_index_list)
