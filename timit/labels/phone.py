#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make phone-level target labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import numpy as np
from tqdm import tqdm
import pickle

from utils.labels.phone import Phone2idx
from utils.util import mkdir_join
from timit.util import map_phone2phone


def read_phone(label_paths, label_type, run_root_path, model,
               save_map_file=False, save_path=None, stdout_transcript=False):
    """Read phone transcript.
    Args:
        label_paths (list): list of paths to label files
        label_type (string): phone39 or phone48 or phone61
        run_root_path (string): path to make.sh
        model (string): ctc or attention
        save_map_file (bool, optional): if True, save the mapping file
        save_path (string, optional): path to save labels. If None, don't save labels
        stdout_transcript (bool, optional): if True, print transcripts to standard output
    """
    if label_type not in ['phone39', 'phone48', 'phone61']:
        raise TypeError('data_type is "phone39" or "phone48" or "phone61".')
    if model not in ['ctc', 'attention']:
        raise TypeError('model must be ctc or attention.')

    # Make the mapping file
    phone2phone_map_file_path = join(run_root_path, 'labels/phone2phone.txt')
    phone2idx_map_file_path = mkdir_join(
        run_root_path, 'labels', 'mapping_files', model, label_type + '.txt')
    phone_set = set([])
    with open(phone2phone_map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[1] != 'nan':
                if label_type == 'phone61':
                    phone_set.add(line[0])
                elif label_type == 'phone48':
                    phone_set.add(line[1])
                elif label_type == 'phone39':
                    phone_set.add(line[2])
            else:
                # Ignore "q" if phone39 or phone48
                if label_type == 'phone61':
                    phone_set.add(line[0])

    # Save mapping file
    if save_map_file:
        with open(phone2idx_map_file_path, 'w') as f:
            if model == 'ctc':
                for index, phone in enumerate(sorted(list(phone_set))):
                    f.write('%s  %s\n' % (phone, str(index)))
            elif model == 'attention':
                for index, phone in enumerate(['<', '>'] + sorted(list(phone_set))):
                    f.write('%s  %s\n' % (phone, str(index)))

    phone2idx = Phone2idx(map_file_path=phone2idx_map_file_path)

    print('===> Reading & Saving target labels...')
    label_num_dict = {}
    for label_path in tqdm(label_paths):
        speaker = label_path.split('/')[-2]
        utt_index = basename(label_path).split('.')[0]

        phone_list = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                # start_frame = line[0]
                # end_frame = line[1]
                phone_list.append(line[2])

        # Map from 61 phones to the corresponding phones
        phone_list = map_phone2phone(phone_list, label_type,
                                     phone2phone_map_file_path)

        # Convert from phone to index
        if model == 'attention':
            phone_list = ['<'] + phone_list + ['>']  # add <SOS> & <EOS>

        if stdout_transcript:
            print(' '.join(phone_list))

        if save_path is not None:
            index_list = phone2idx(phone_list)

            # Save phone labels as npy file
            np.save(mkdir_join(save_path, speaker + '_' +
                               utt_index + '.npy'), index_list)

            # Count label number
            label_num_dict[speaker + '_' + utt_index] = len(index_list)

    if save_path is not None:
        # Save the label number dictionary
        with open(join(save_path, 'label_num.pickle'), 'wb') as f:
            pickle.dump(label_num_dict, f)
