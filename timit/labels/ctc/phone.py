#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make label for CTC model (TIMIT corpus)."""

import os
import numpy as np
from tqdm import tqdm

from prepare_path import Prepare
from utils.labels.phone import phone2num
from util import map_phone2phone


def read_phone(label_paths, label_type, save_path=None):
    """Read phone transcript.
    Args:
        label_paths: list of paths to label files
        label_type: phone39 or phone48 or phone61
        save_path: path to save labels. If None, don't save labels
    """
    if label_type not in ['phone39', 'phone48', 'phone61']:
        raise ValueError('Error: data_type is "phone39" or "phone48" or "phone61".')

    print('===> Reading & Saving target labels...')
    prep = Prepare()
    p2p_map_file_path = os.path.join(prep.run_root_path, 'labels/phone2phone.txt')
    for label_path in tqdm(label_paths):
        speaker_name = label_path.split('/')[-2]
        file_name = label_path.split('/')[-1].split('.')[0]
        save_file_name = speaker_name + '_' + file_name + '.npy'

        phone_list = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                # start_frame = line[0]
                # end_frame = line[1]
                phone_list.append(line[2])

        # map from 61 phones to the corresponding phones
        phone_list = map_phone2phone(phone_list, label_type, p2p_map_file_path)

        # make the mapping file
        p2n_map_file_path = os.path.join(
            prep.run_root_path, 'labels/ctc/phone2num_' + label_type[5:7] + '.txt')
        if not os.path.isfile(p2n_map_file_path):
            phone_set = set([])
            with open(p2p_map_file_path, 'r') as f:
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
                        # ignore "q" if phone39 or phone48
                        if label_type == 'phone61':
                            phone_set.add(line[0])

            # save mapping file
            with open(p2n_map_file_path, 'w') as f:
                for index, phone in enumerate(sorted(list(phone_set))):
                    f.write('%s  %s\n' % (phone, str(index)))

        # convert from phone to number
        phone_list = phone2num(phone_list, p2n_map_file_path)

        if save_path is not None:
            # save phone labels as npy file
            np.save(os.path.join(save_path, save_file_name), phone_list)
