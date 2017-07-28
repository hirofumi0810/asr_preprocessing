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
             save_path=None,
             train_mean_male=None, train_mean_female=None,
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

    # Read each HTK file
    print('===> Reading HTK files...')
    input_data_dict = {}
    input_data_list_male, input_data_list_female = [], []
    total_frame_num_dict = {}
    total_frame_num_male, total_frame_num_female = 0, 0

    for htk_path in tqdm(htk_paths):
        utterance_name = basename(htk_path).split('.')[0]
        speaker_index, book_index, utterance_index = utterance_name.split('-')

        # Read HTK
        input_data_utt = read_htk_utt(htk_path)
        if speaker_index not in input_data_dict.keys():
            input_data_dict[speaker_index] = {}
        if speaker_index not in total_frame_num_dict.keys():
            total_frame_num_dict[speaker_index] = 0
        input_data_dict[speaker_index][utterance_name] = input_data_utt
        total_frame_num_dict[speaker_index] += input_data_utt.shape[0]

        if speaker_gender_dict[speaker_index] == 'M':
            input_data_list_male.append(input_data_utt)
            total_frame_num_male += input_data_utt.shape[0]
        elif speaker_gender_dict[speaker_index] == 'F':
            input_data_list_female.append(input_data_utt)
            total_frame_num_female += input_data_utt.shape[0]

    feature_dim = input_data_list_male[0].shape[1]
    if is_training:
        # Compute global mean per gender
        print('===> Computing global mean over the training set...')
        print('=====> male...')
        train_mean_male = np.zeros((feature_dim,), dtype=np.float64)
        for input_data_utt in tqdm(input_data_list_male):
            train_mean_male += np.sum(input_data_utt, axis=0)
        train_mean_male /= total_frame_num_male

        print('=====> female...')
        train_mean_female = np.zeros((feature_dim,), dtype=np.float64)
        for input_data_utt in tqdm(input_data_list_female):
            train_mean_female += np.sum(input_data_utt, axis=0)
        train_mean_female /= total_frame_num_female

        # Compute global standard deviation per gender
        print('===> Computing global std over the training set...')
        print('=====> male...')
        train_std_male = np.zeros((feature_dim,), dtype=np.float64)
        for input_data_utt in tqdm(input_data_list_male):
            train_std_male += np.sum(
                np.abs(input_data_utt - train_mean_male) ** 2, axis=0)
        train_std_male = np.sqrt(train_std_male / (total_frame_num_male - 1))

        print('=====> female...')
        train_std_female = np.zeros((feature_dim,), dtype=np.float64)
        for input_data_utt in tqdm(input_data_list_female):
            train_std_female += np.sum(
                np.abs(input_data_utt - train_mean_female) ** 2, axis=0)
        train_std_female = np.sqrt(
            train_std_female / (total_frame_num_female - 1))

    # Normalization
    print('===> Normalization...')
    frame_num_dict = {}
    for speaker_index, utterance_dict in tqdm(input_data_dict.items()):
        if normalize == 'speaker':
            # Compute mean & std per speaker
            input_data_concat_speaker = np.zeros(
                (total_frame_num_dict[speaker_index], feature_dim),
                dtype=np.float64)
            frame_offset = 0
            for input_data_utt in utterance_dict.values():
                frame_num_utt = input_data_utt.shape[0]
                input_data_concat_speaker[frame_offset:frame_offset +
                                          frame_num_utt] = input_data_utt
                frame_offset += frame_num_utt
            speaker_mean = np.mean(input_data_concat_speaker, axis=0)
            speaker_std = np.std(input_data_concat_speaker, axis=0)

        for utterance_name, input_data_utt in utterance_dict.items():
            if normalize == 'utterance':
                # Normalize by mean & std per utterance
                utt_mean = np.mean(input_data_utt, axis=0,
                                   dtype=np.float64)
                utt_std = np.std(input_data_utt, axis=0, dtype=np.float64)
                input_data_utt = (input_data_utt - utt_mean) / utt_std
                input_data_dict[speaker_index][utterance_name] = input_data_utt

            elif normalize == 'speaker':
                # Normalize by mean & std per speaker
                input_data_utt -= speaker_mean
                input_data_utt /= speaker_std
                input_data_dict[speaker_index][utterance_name] = input_data_utt

            elif normalize == 'global':
                # Normalize by mean & std over the training set
                if speaker_gender_dict[speaker_index] == 'M':
                    input_data_utt -= train_mean_male
                    input_data_utt /= train_std_male
                elif speaker_gender_dict[speaker_index] == 'F':
                    input_data_utt -= train_mean_female
                    input_data_utt /= train_std_female
                input_data_dict[speaker_index][utterance_name] = input_data_utt

            else:
                raise ValueError(
                    'normalize is "utterance" or "speaker" or "global".')

            if save_path is not None:
                # Save input data
                mkdir_join(save_path, speaker_index)
                input_data_save_path = join(
                    save_path, speaker_index, utterance_name + '.npy')

                np.save(input_data_save_path, input_data_utt)
                frame_num_dict[utterance_name] = input_data_utt.shape[0]

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

    if save_path is not None:
        # Save the frame number dictionary
        print('===> Saving : frame_num.pickle')
        with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
            pickle.dump(frame_num_dict, f)

    return train_mean_male, train_mean_female, train_std_male, train_std_female
