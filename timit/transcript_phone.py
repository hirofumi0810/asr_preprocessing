#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make phone-level target labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
from tqdm import tqdm

from utils.util import mkdir_join
from timit.util import map_phone2phone


def read_phone(label_paths, vocab_file_save_path, save_vocab_file=False):
    """Read phone transcript.
    Args:
        label_paths (list): list of paths to label files
        vocab_file_save_path (string): path to vocabulary files
        save_vocab_file (bool, optional): if True, save vocabulary files
    Returns:
        text_dict (dict):
            key (string) => utterance name
            value (list) => list of [trans_phone61, trans_phone48, trans_phone39]
    """
    # Make the mapping file (from phone to index)
    phone2phone_map_file_path = join(
        vocab_file_save_path, '../phone2phone.txt')
    phone61_set, phone48_set, phone39_set = set([]), set([]), set([])
    with open(phone2phone_map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[1] != 'nan':
                phone61_set.add(line[0])
                phone48_set.add(line[1])
                phone39_set.add(line[2])
            else:
                # Ignore "q" if phone39 or phone48
                phone61_set.add(line[0])

    phone61_to_idx_map_file_path = mkdir_join(
        vocab_file_save_path, 'phone61.txt')
    phone48_to_idx_map_file_path = mkdir_join(
        vocab_file_save_path, 'phone48.txt')
    phone39_to_idx_map_file_path = mkdir_join(
        vocab_file_save_path, 'phone39.txt')

    # Save mapping file
    if save_vocab_file:
        with open(phone61_to_idx_map_file_path, 'w') as f:
            for phone in sorted(list(phone61_set)):
                f.write('%s\n' % phone)
        with open(phone48_to_idx_map_file_path, 'w') as f:
            for phone in sorted(list(phone48_set)):
                f.write('%s\n' % phone)
        with open(phone39_to_idx_map_file_path, 'w') as f:
            for phone in sorted(list(phone39_set)):
                f.write('%s\n' % phone)

    print('===> Reading target labels...')
    trans_dict = {}
    for label_path in tqdm(label_paths):
        speaker = label_path.split('/')[-2]
        utt_index = basename(label_path).split('.')[0]
        utt_name = speaker + '_' + utt_index

        phone61_list = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                # start_frame = line[0]
                # end_frame = line[1]
                phone61_list.append(line[2])

        # Map from 61 phones to the corresponding phones
        phone48_list = map_phone2phone(phone61_list, 'phone48',
                                       phone2phone_map_file_path)
        phone39_list = map_phone2phone(phone61_list, 'phone39',
                                       phone2phone_map_file_path)

        # Convert to string
        trans_phone61 = ' '.join(phone61_list)
        trans_phone48 = ' '.join(phone48_list)
        trans_phone39 = ' '.join(phone39_list)

        # for debug
        # print(trans_phone61)
        # print(trans_phone48)
        # print(trans_phone39)
        # print('-----')

        trans_dict[utt_name] = [trans_phone61, trans_phone48, trans_phone39]

    return trans_dict
