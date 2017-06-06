#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (Switchboard corpus)."""

from os.path import join, basename
import pickle
import numpy as np
from tqdm import tqdm

from utils.util import mkdir
from utils.inputs.segmentation import segment_htk, global_mean, global_std


def read_htk(htk_paths, speaker_dict, normalize, is_training,
             save_path=None, train_mean=None, train_std=None):
    """Read HTK files.
    Args:
        htk_paths: list of paths to HTK files
        speaker_dict: dictionary of speakers
            key => speaker name
            value => dictionary of utterance information of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
        normalize: global => normalize input data by global mean & std of train data
                   speaker => normalize data by mean & std per speaker
                   None => no normalization
        is_training: training or not
        save_path: path to save npy files
        train_mean: global mean over train data
        train_std: global standard deviation over train data
    Returns:
        train_mean: global mean over train data
        train_std: global standard deviation over train data
    """
    # Read each HTK file
    print('===> Reading HTK files...')
    input_data_dict_list = []
    train_mean_list = []
    train_std_list = []
    total_frame_num_list = []
    for htk_path in tqdm(htk_paths):
        speaker_name = basename(htk_path).split('.')[0]
        # e.g. sw04771A => sw4771A (LDC97S62)
        speaker_name = speaker_name.replace('sw0', 'sw')
        # e.g. sw_4771A => sw4771A (eval2000)
        speaker_name = speaker_name.replace('sw_', 'sw')

        return_tuple = segment_htk(htk_path,
                                   speaker_name,
                                   speaker_dict[speaker_name],
                                   normalize=normalize,
                                   sil_duration=50)
        input_data_dict, train_mean, train_std, total_frame_num = return_tuple

        input_data_dict_list.append(input_data_dict)
        train_mean_list.append(train_mean)
        train_std_list.append(train_std)
        total_frame_num_list.append(total_frame_num)

    if is_training:
        # Compute global mean
        print('===> Computing global mean over train data...')
        train_global_mean = global_mean(train_mean_list,
                                        total_frame_num_list)

        # Compute global standard deviation
        print('===> Computing global std over train data...')
        train_global_std = global_std(input_data_dict_list,
                                      total_frame_num_list,
                                      train_global_mean)

        if save_path is not None:
            # Save global mean & std
            statistics_save_path = '/'.join(save_path.split('/')[:-1])
            np.save(join(statistics_save_path,
                         'train_mean.npy'), train_global_mean)
            np.save(join(statistics_save_path, 'train_std.npy'), train_global_std)

    if save_path is not None:
        # Save input data
        print('===> Saving input data...')
        frame_num_dict = {}
        for input_data_dict in tqdm(input_data_dict_list):
            for key, input_data_utt in input_data_dict.items():
                # Normalize by global mean & std over train data
                if normalize == 'global':
                    input_data_utt -= train_global_mean
                    input_data_utt /= train_global_std

                speaker_name, utt_index = key.split('_')
                mkdir(save_path, speaker_name)
                input_data_save_path = join(
                    save_path, speaker_name, key + '.npy')

                np.save(input_data_save_path, input_data_utt)
                frame_num_dict[key] = input_data_utt.shape[0]

        # Save the frame number dictionary
        frame_num_dict_save_path = '/'.join(save_path.split('/')[:-1])
        print('===> Saving : frame_num.pickle')
        with open(join(frame_num_dict_save_path, 'frame_num.pickle'), 'wb') as f:
            pickle.dump(frame_num_dict, f)

    return train_mean, train_std
