#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import pickle
import numpy as np
from tqdm import tqdm

from utils.inputs.segmentation import read_htk as read_htk_utt


def read_htk(htk_paths, normalize, is_training, save_path=None,
             train_mean=None, train_std=None):
    """Read HTK files.
    Args:
        htk_paths: list of paths to HTK files
        save_path: path to save npy files
        normalize: if True, normalize all data by mean & std of train data
        is_training: training or not
        train_mean: mean over train data
        train_std: standard deviation over train data
    Returns:
        train_mean: mean over train data, return only when training
        train_std: standard deviation over train data, return only when training
    """
    # Read each HTK file
    print('===> Reading HTK files...')
    input_data_list = []
    for htk_path in tqdm(htk_paths):
        input_data_list.append(read_htk_utt(htk_path))

    if normalize and is_training:
        # Count total frame num
        total_frame_num = 0
        for input_data in input_data_list:
            total_frame_num += input_data.shape[0]

        # Compute global mean & std
        print('===> Computing global mean & std over train data...')
        frame_offset = 0
        feature_dim = input_data_list[0].shape[1]
        train_data = np.empty((total_frame_num, feature_dim))
        for input_data_utt in tqdm(input_data_list):
            frame_num_utt = input_data_utt.shape[0]
            train_data[frame_offset:frame_offset +
                       frame_num_utt] = input_data_utt
            frame_offset += frame_num_utt
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)

        if save_path is not None:
            # Save global mean & std
            np.save(join(save_path, 'train_mean.npy'), train_mean)
            np.save(join(save_path, 'train_std.npy'), train_std)

    if save_path is not None:
        # Save input data as npy files
        print('===> Saving input data...')
        frame_num_dict = {}
        for input_data, htk_path in zip(tqdm(input_data_list), htk_paths):
            input_data_save_name = basename(htk_path).split('.')[0] + '.npy'
            input_data_save_path = join(save_path, input_data_save_name)

            # Normalize by global mean & std over train data
            if normalize:
                input_data = (input_data - train_mean) / train_std

            np.save(input_data_save_path, input_data)
            frame_num_dict[basename(htk_path).split('.')[
                0]] = input_data.shape[0]

        # Save a frame number dictionary
        with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
            print('===> Saving : frame_num.pickle')
            pickle.dump(frame_num_dict, f)

    return train_mean, train_std
