#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import pickle
import numpy as np
from tqdm import tqdm

from utils.util import mkdir_join
from utils.inputs.segmentation import read_htk as read_htk_utt


def read_htk(htk_paths, normalize, is_training, speaker_gender_dict,
             save_path=None, train_mean_male=None, train_mean_female=None,
             train_std_male=None, train_std_female=None):
    """Read HTK files.
    Args:
        htk_paths: list of paths to HTK files
            key => speaker index
            value => dictionary of utterance information of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
        normalize: global => normalize input data by global mean & std over
                             the training set per gender
                   speaker => normalize input data by mean & std per speaker
                   utterance => normalize input data by mean & std per
                                utterancet data by mean & std per
                                utterance
        is_training: training or not
        speaker_gender_dict: A dictionary of speakers' gender information
            key => speaker_index
            value => F or M
        save_path: path to save npy files
        train_mean_male: global mean of male over the training set
        train_mean_female: global mean of female over the training set
        train_std_male: global standard deviation of female over the training
            set
        train_std_female: global standard deviation of female over the training
            set
    Returns:
        train_mean_male: global mean of male over the training set
        train_mean_female: global mean of female over the training set
        train_std_male: global standard deviation of male over the training set
        train_std_female: global standard deviation of female over the training
            set
    """
    if not is_training:
        if train_mean_male is None or train_mean_female is None or train_std_male is None or train_std_female is None:
            raise ValueError('Set mean & std computed in the training set.')
    if normalize not in ['global', 'speaker', 'utterance']:
        raise ValueError(
            'normalize is "utterance" or "speaker" or "global".')

    # Divide all htk paths into speakers
    print('===> Reading HTK files...')
    htk_path_dict = {}
    htk_path_list_male, htk_path_list_female = [], []
    total_frame_num_dict = {}
    total_frame_num_male, total_frame_num_female = 0, 0
    speaker_mean_dict, speaker_std_dict = {}, {}
    for i, htk_path in enumerate(tqdm(htk_paths)):
        utterance_name = basename(htk_path).split('.')[0]
        speaker_index = utterance_name.split('-')[0]
        if speaker_index not in htk_path_dict.keys():
            htk_path_dict[speaker_index] = []
        htk_path_dict[speaker_index].append(htk_path)

        if is_training:
            # Read each HTK file
            input_data_utt = read_htk_utt(htk_path)
            input_data_utt_sum = np.sum(input_data_utt, axis=0)

            if i == 0:
                # Initialize global statistics
                feature_dim = input_data_utt.shape[1]
                train_mean_male = np.zeros((feature_dim,), dtype=np.float64)
                train_mean_female = np.zeros((feature_dim,), dtype=np.float64)
                train_std_male = np.zeros((feature_dim,), dtype=np.float64)
                train_std_female = np.zeros((feature_dim,), dtype=np.float64)

            # For computing global mean
            if speaker_gender_dict[speaker_index] == 'M':
                htk_path_list_male.append(input_data_utt)
                train_mean_male += input_data_utt_sum
                total_frame_num_male += input_data_utt.shape[0]
            elif speaker_gender_dict[speaker_index] == 'F':
                htk_path_list_female.append(input_data_utt)
                train_mean_female += np.sum(input_data_utt, axis=0)
                total_frame_num_female += input_data_utt.shape[0]

            # For computing speaker mean
            if normalize == 'speaker':
                if speaker_index not in total_frame_num_dict.keys():
                    total_frame_num_dict[speaker_index] = 0
                    # Initialize speaker statistics
                    speaker_mean_dict[speaker_index] = np.zeros(
                        (feature_dim,), dtype=np.float64)
                    speaker_std_dict[speaker_index] = np.zeros(
                        (feature_dim,), dtype=np.float64)
                speaker_mean_dict[speaker_index] += input_data_utt_sum
                total_frame_num_dict[speaker_index] += input_data_utt.shape[0]

    if is_training:
        print('===> Computing global mean & stddev...')
        # Compute global mean per gender
        train_mean_male /= total_frame_num_male
        train_mean_female /= total_frame_num_female

        for speaker_index, htk_paths_speaker in tqdm(htk_path_dict.items()):
            if normalize == 'speaker':
                # Compute speaker mean
                speaker_mean_dict[speaker_index] /= total_frame_num_dict[speaker_index]

            for htk_path in htk_paths_speaker:
                utterance_name = basename(htk_path).split('.')[0]
                speaker_index = utterance_name.split('-')[0]

                # Read each HTK file
                input_data_utt = read_htk_utt(htk_path)

                # For computeing global stddev
                if speaker_gender_dict[speaker_index] == 'M':
                    train_std_male += np.sum(
                        np.abs(input_data_utt - train_mean_male) ** 2, axis=0)
                elif speaker_gender_dict[speaker_index] == 'F':
                    train_std_female += np.sum(
                        np.abs(input_data_utt - train_mean_female) ** 2, axis=0)

                if normalize == 'speaker':
                    # For computeing speaker stddev
                    speaker_std_dict[speaker_index] += np.sum(
                        np.abs(input_data_utt - speaker_mean_dict[speaker_index]) ** 2, axis=0)

            if normalize == 'speaker':
                # Compute speaker stddev
                speaker_std_dict[speaker_index] = np.sqrt(
                    speaker_std_dict[speaker_index] / (total_frame_num_dict[speaker_index] - 1))

        # Compute global stddev per gender
        train_std_male = np.sqrt(train_std_male / (total_frame_num_male - 1))
        train_std_female = np.sqrt(
            train_std_female / (total_frame_num_female - 1))

        if save_path is not None:
            # Save global mean & std per gender
            np.save(join(save_path, 'train_mean_male.npy'),
                    train_mean_male)
            np.save(join(save_path, 'train_mean_female.npy'),
                    train_mean_female)
            np.save(join(save_path, 'train_std_male.npy'),
                    train_std_male)
            np.save(join(save_path, 'train_std_female.npy'),
                    train_std_female)

    # Normalization
    print('===> Normalization...')
    frame_num_dict = {}
    for speaker_index, htk_paths_speaker in tqdm(htk_path_dict.items()):
        for htk_path in htk_paths_speaker:
            utterance_name = basename(htk_path).split('.')[0]
            speaker_index = utterance_name.split('-')[0]

            # Read each HTK file
            input_data_utt = read_htk_utt(htk_path)

            if normalize == 'utterance' and is_training:
                # Normalize by mean & std per utterance
                utt_mean = np.mean(input_data_utt, axis=0, dtype=np.float64)
                utt_std = np.std(input_data_utt, axis=0, dtype=np.float64)
                input_data_utt = (input_data_utt - utt_mean) / utt_std

            elif normalize == 'speaker' and is_training:
                # Normalize by mean & std per speaker
                input_data_utt -= speaker_mean_dict[speaker_index]
                input_data_utt /= speaker_std_dict[speaker_index]

            else:
                # Normalize by mean & std over the training set per gender
                if speaker_gender_dict[speaker_index] == 'M':
                    input_data_utt -= train_mean_male
                    input_data_utt /= train_std_male
                elif speaker_gender_dict[speaker_index] == 'F':
                    input_data_utt -= train_mean_female
                    input_data_utt /= train_std_female

            if save_path is not None:
                # Save input data
                mkdir_join(save_path, speaker_index)
                input_data_save_path = join(
                    save_path, speaker_index, utterance_name + '.npy')
                np.save(input_data_save_path, input_data_utt)
                frame_num_dict[utterance_name] = input_data_utt.shape[0]

    if save_path is not None:
        # Save the frame number dictionary
        print('===> Saving : frame_num.pickle')
        with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
            pickle.dump(frame_num_dict, f)

    return train_mean_male, train_mean_female, train_std_male, train_std_female
