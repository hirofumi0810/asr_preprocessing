#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import pickle
import numpy as np
from tqdm import tqdm

from utils.util import mkdir_join
from utils.inputs.segmentation import segment_htk, global_mean, global_std


def read_htk(htk_paths, speaker_dict, normalize, is_training, save_path=None,
             train_mean_male=None, train_mean_female=None,
             train_std_male=None, train_std_female=None):
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
        train_mean_male: global mean of male over train data
        train_mean_female: global mean of female over train data
        train_std_male: global standard deviation of male over train data
        train_std_female: global standard deviation of female over train data
    """
    # Load each HTK file
    print('===> Loading HTK files...')
    input_data_dict_list_male, total_frame_num_list_male = [], []
    train_mean_list_male, train_std_list_male = [], []
    input_data_dict_list_female, total_frame_num_list_female = [], []
    train_mean_list_female, train_std_list_female = [], []
    for htk_path in tqdm(htk_paths):
        speaker_name = basename(htk_path).split('.')[0]
        return_tuple = segment_htk(htk_path,
                                   speaker_name,
                                   speaker_dict[speaker_name],
                                   normalize=normalize,
                                   sil_duration=50)
        input_data_dict, train_mean, train_std, total_frame_num = return_tuple

        if speaker_name[3] == 'M':
            input_data_dict_list_male.append(input_data_dict)
            train_mean_list_male.append(train_mean)
            train_std_list_male.append(train_std)
            total_frame_num_list_male.append(total_frame_num)
        elif speaker_name[3] == 'F':
            input_data_dict_list_female.append(input_data_dict)
            train_mean_list_female.append(train_mean)
            train_std_list_female.append(train_std)
            total_frame_num_list_female.append(total_frame_num)

    if is_training:
        # Compute global mean (each gender)
        print('===> Computing global mean over train data...')
        print('=====> male...')
        train_mean_male = global_mean(train_mean_list_male,
                                      total_frame_num_list_male)
        print('=====> female...')
        train_mean_female = global_mean(train_mean_list_female,
                                        total_frame_num_list_female)

        # Compute global standard deviation (each gender)
        print('===> Computing global std over train data...')
        print('=====> male...')
        train_std_male = global_std(input_data_dict_list_male,
                                    total_frame_num_list_male,
                                    train_mean_male)
        print('=====> female...')
        train_std_female = global_std(input_data_dict_list_female,
                                      total_frame_num_list_female,
                                      train_mean_female)

        if save_path is not None:
            # Save global mean & std (each gender)
            statistics_save_path = '/'.join(save_path.split('/')[:-1])
            np.save(join(statistics_save_path, 'train_mean_male.npy'),
                    train_mean_male)
            np.save(join(statistics_save_path, 'train_mean_female.npy'),
                    train_mean_female)
            np.save(join(statistics_save_path, 'train_std_male.npy'),
                    train_std_male)
            np.save(join(statistics_save_path, 'train_std_female.npy'),
                    train_std_female)

    if save_path is not None:
        # Save input data
        print('===> Saving input data...')
        frame_num_dict = {}
        for input_data_dict in tqdm(input_data_dict_list_male + input_data_dict_list_female):
            for key, input_data_utt in input_data_dict.items():
                speaker_name, utt_index = key.split('_')

                # Normalize by global mean & std over train data (each gender)
                if normalize == 'global':
                    if speaker_name[3] == 'M':
                        input_data_utt -= train_mean_male
                        input_data_utt /= train_std_male
                    elif speaker_name[3] == 'F':
                        input_data_utt -= train_mean_female
                        input_data_utt /= train_std_female

                mkdir_join(save_path, speaker_name)
                input_data_save_path = join(
                    save_path, speaker_name, key + '.npy')

                np.save(input_data_save_path, input_data_utt)
                frame_num_dict[key] = input_data_utt.shape[0]

        # Save the frame number dictionary
        frame_num_dict_save_path = '/'.join(save_path.split('/')[:-1])
        print('===> Saving : frame_num.pickle')
        with open(join(frame_num_dict_save_path, 'frame_num.pickle'), 'wb') as f:
            pickle.dump(frame_num_dict, f)

    return train_mean_male, train_mean_female, train_std_male, train_std_female
