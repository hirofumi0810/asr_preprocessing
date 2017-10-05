#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make phone-level target labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import numpy as np
from tqdm import tqdm

from utils.labels.phone import Phone2idx
from utils.util import mkdir_join
from timit.util import map_phone2phone

SOS = '<'
EOS = '>'


def read_phone(label_paths, run_root_path, save_map_file=False,
               ctc_phone61_save_path=None, ctc_phone48_save_path=None,
               ctc_phone39_save_path=None, att_phone61_save_path=None,
               att_phone48_save_path=None, att_phone39_save_path=None):
    """Read phone transcript.
    Args:
        label_paths (list): list of paths to label files
        run_root_path (string): path to make.sh
        save_map_file (bool, optional): if True, save the mapping file
        ctc_phone61_save_path (string, optional): path to save labels for the
            CTC models (phone61). If None, don't save labels
        ctc_phone48_save_path (string, optional): path to save labels for the
            CTC models (phone48). If None, don't save labels
        ctc_phone39_save_path (string, optional): path to save labels for the.
            CTC models (phone39). If None, don't save labels
        att_phone61_save_path (string, optional): path to save labels for the
            Attention-based models (phone61). If None, don't save labels
        att_phone48_save_path (string, optional): path to save labels for the
            Attention-based models (phone48). If None, don't save labels
        att_phone39_save_path (string, optional): path to save labels for the.
            Attention-based models (phone39). If None, don't save labels
    """
    # Make the mapping file (from phone to index)
    phone2phone_map_file_path = join(run_root_path, 'labels/phone2phone.txt')
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
        run_root_path, 'labels', 'mapping_files', 'phone61.txt')
    phone48_to_idx_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', 'phone48.txt')
    phone39_to_idx_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', 'phone39.txt')

    # Save mapping file
    if save_map_file:
        with open(phone61_to_idx_map_file_path, 'w') as f:
            for i, phone in enumerate(sorted(list(phone61_set)) + [SOS, EOS]):
                f.write('%s  %s\n' % (phone, str(i)))
        with open(phone48_to_idx_map_file_path, 'w') as f:
            for i, phone in enumerate(sorted(list(phone48_set)) + [SOS, EOS]):
                f.write('%s  %s\n' % (phone, str(i)))
        with open(phone39_to_idx_map_file_path, 'w') as f:
            for i, phone in enumerate(sorted(list(phone39_set)) + [SOS, EOS]):
                f.write('%s  %s\n' % (phone, str(i)))

    phone61_to_idx = Phone2idx(map_file_path=phone61_to_idx_map_file_path)
    phone48_to_idx = Phone2idx(map_file_path=phone48_to_idx_map_file_path)
    phone39_to_idx = Phone2idx(map_file_path=phone39_to_idx_map_file_path)

    print('===> Reading & Saving target labels...')
    for label_path in tqdm(label_paths):
        speaker = label_path.split('/')[-2]
        utt_index = basename(label_path).split('.')[0]

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

        # for debug
        # print(' '.join(phone61_list))
        # print(' '.join(phone48_list))
        # print(' '.join(phone39_list))
        # print('-----')

        # Convert from phone to index
        ctc_phone61_index_list = phone61_to_idx(phone61_list)
        ctc_phone48_index_list = phone48_to_idx(phone48_list)
        ctc_phone39_index_list = phone39_to_idx(phone39_list)
        att_phone61_index_list = phone61_to_idx(
            [SOS] + phone61_list + [EOS])
        att_phone48_index_list = phone48_to_idx(
            [SOS] + phone48_list + [EOS])
        att_phone39_index_list = phone39_to_idx(
            [SOS] + phone39_list + [EOS])

        # Save phone labels as npy file
        save_file_name = speaker + '_' + utt_index + '.npy'
        if ctc_phone61_save_path is not None:
            np.save(mkdir_join(ctc_phone61_save_path,
                               save_file_name), ctc_phone61_index_list)
        if att_phone61_save_path is not None:
            np.save(mkdir_join(att_phone61_save_path,
                               save_file_name), att_phone61_index_list)

        if ctc_phone48_save_path is not None:
            np.save(mkdir_join(ctc_phone48_save_path,
                               save_file_name), ctc_phone48_index_list)
        if att_phone48_save_path is not None:
            np.save(mkdir_join(att_phone48_save_path, save_file_name),
                    att_phone48_index_list)

        if ctc_phone39_save_path is not None:
            np.save(mkdir_join(ctc_phone39_save_path,
                               save_file_name), ctc_phone39_index_list)
        if att_phone39_save_path is not None:
            np.save(mkdir_join(att_phone39_save_path,
                               save_file_name), att_phone39_index_list)
