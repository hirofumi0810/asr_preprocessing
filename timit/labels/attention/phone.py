#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make label for Attention model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import numpy as np
from tqdm import tqdm

from utils.labels.phone import phone2num
from util import map_phone2phone


def read_phone(label_paths, label_type, run_root_path, save_path=None):
    """Read phone transcript.
    Args:
        label_paths: list of paths to label files
        label_type: phone39 or phone48 or phone61
        run_root_path: path to make.sh
        save_path: path to save labels. If None, don't save labels
    """
    if label_type not in ['phone39', 'phone48', 'phone61']:
        raise ValueError(
            'data_type is "phone39" or "phone48" or "phone61".')

    print('===> Reading & Saving target labels...')
    phone2phone_map_file_path = join(run_root_path, 'labels/phone2phone.txt')
    phone2num_map_file_path = join(
        run_root_path,
        'labels/attention/phone2num_' + label_type[5:7] + '.txt')
    for label_path in tqdm(label_paths):
        speaker_name = label_path.split('/')[-2]
        file_name = label_path.split('/')[-1].split('.')[0]
        save_file_name = speaker_name + '_' + file_name + '.npy'

        phone_list = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                # Start_frame = line[0]
                # end_frame = line[1]
                phone_list.append(line[2])

        # Map from 61 phones to 31 phones
        phone_list = map_phone2phone(phone_list, label_type,
                                     phone2phone_map_file_path)

        # Make the mapping file
        if not isfile(phone2num_map_file_path):
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
            with open(phone2num_map_file_path, 'w') as f:
                for index, phone in enumerate(sorted(list(phone_set))):
                    f.write('%s  %s\n' % (phone, str(index)))
                f.write('%s  %s\n' % ('<', str(index + 1)))
                f.write('%s  %s\n' % ('>', str(index + 2)))

        # Convert from phone to number
        phone_list = ['<'] + phone_list + ['>']  # add SOS & EOS
        phone_list = phone2num(phone_list, phone2num_map_file_path)

        if save_path is not None:
            # Save phone labels as npy file
            np.save(join(save_path, save_file_name), phone_list)
