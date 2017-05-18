#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (Switchboard corpus).
   Normalize over training data.
"""

# import gc
import os
import pickle
import numpy as np
import functools
from tqdm import tqdm

from utils.util import mkdir
from utils.inputs.read_htk import read
np.seterr(all='ignore')


def read_htk(htk_paths, speaker_dict, normalize, is_training, save_path=None, train_mean=None, train_std=None):
    """Read HTK files.
    Args:
        htk_paths: list of paths to HTK files
        speaker_dict: dictionary of speakers
            key => speaker name
            value => dictionary of utterance infomation of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
        normalize: if True, normalize all data by mean & std of train data
        is_training: training or not
        save_path: path to save npy files
        train_mean: global mean over train data
        train_std: global standard deviation over train data
    Returns:
        train_mean: global mean over train data
        train_std: global standard deviation over train data
    """
    # load each HTK file
    print('===> Reading HTK files...')
    input_data_dict_list = []
    for htk_path in tqdm(htk_paths):
        speaker_name = os.path.basename(htk_path).split('.')[0]
        # e.g. sw04771A => sw4771A (LDC97S62)
        speaker_name = speaker_name.replace('sw0', 'sw')
        # e.g. sw_4771A => sw4771A (eval2000)
        speaker_name = speaker_name.replace('sw_', 'sw')

        input_data_dict_list.append(
            read_each_htk(htk_path, speaker_name, speaker_dict[speaker_name]))

    # merge all dicts
    print('===> Merge all dictionaries...')
    all_input_data_dict = functools.reduce(lambda first, second: dict(first, **second),
                                           input_data_dict_list)

    if normalize and is_training:
        # count total frame num
        total_frame_num = 0
        for input_data_utt in all_input_data_dict.values():
            total_frame_num += input_data_utt.shape[0]

        # compute global mean & std
        print('===> Computing global mean & std over train data...')
        frame_offset = 0
        feature_dim = 123
        train_data = np.empty((total_frame_num, feature_dim))
        for input_data_utt in tqdm(all_input_data_dict.values()):
            frame_num_utt = input_data_utt.shape[0]
            train_data[frame_offset:frame_offset +
                       frame_num_utt] = input_data_utt
            frame_offset += frame_num_utt
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)

        if save_path is not None:
            # save global mean & std
            statistics_save_path = '/'.join(save_path.split('/')[:-1])
            np.save(os.path.join(statistics_save_path,
                                 'train_mean.npy'), train_mean)
            np.save(os.path.join(statistics_save_path,
                                 'train_std.npy'), train_std)
            # del train_data
            # gc.collect()

    if save_path is not None:
        # save input data
        print('===> Saving input data...')
        frame_num_dict = {}
        for input_data_save_name, input_data_utt in tqdm(all_input_data_dict.items()):
            # normalize by global mean & std over train data
            if normalize:
                input_data_utt = (input_data_utt - train_mean) / train_std

            speaker_name, utt_index = input_data_save_name.split('_')
            mkdir(os.path.join(save_path, speaker_name))
            input_data_save_path = os.path.join(
                save_path, speaker_name, input_data_save_name + '.npy')

            np.save(input_data_save_path, input_data_utt)
            frame_num_dict[input_data_save_name] = input_data_utt.shape[0]

        # save a frame number dictionary
        frame_num_dict_save_path = '/'.join(save_path.split('/')[:-1])
        with open(os.path.join(frame_num_dict_save_path, 'frame_num.pickle'), 'wb') as f:
            print('===> Saving : frame_num.pickle')
            pickle.dump(frame_num_dict, f)

    return train_mean, train_std


def read_each_htk(htk_path, speaker_name, utterance_dict):
    """Read each HTK file.
    Args:
        htk_path: path to a HTK file
        speaker_name: speaker name
        utterance_dict: dictionary of utterance infomation of each speaker
            key => utterance index
            value => [start_frame, end_frame, transcript]
    Returns:
        input_data_dict:
            key => speaker_name + utterance_index
            value => np.ndarray, (frame_num, feature_dim)
    """
    # print('=====> Reading: ' + os.path.basename(htk_path))

    # read htk
    input_data = read(htk_path)

    # divide into each utterance
    input_data_dict = {}
    total_frame_num = 0
    end_frame_pre = 0
    utt_num = len(utterance_dict.keys())
    utterance_dict_sorted = sorted(utterance_dict.items(), key=lambda x: x[0])
    for index, (utt_index, utt_info) in enumerate(utterance_dict_sorted):
        start_frame, end_frame = utt_info[0], utt_info[1]

        # error check
        if start_frame > end_frame:
            print('Warning: time stamp is reversed. %s' % speaker_name)

        # first utterance
        if index == 0:
            if start_frame >= 50:
                start_frame_extend = start_frame - 50
            else:
                start_frame_extend = 0

            start_frame_next = utterance_dict_sorted[index + 1][1][0]
            if end_frame > start_frame_next:
                print('Warning: utterances are overlapping.')
                print('speaker: %s' % speaker_name)
                print('utt index: %s & %s' %
                      (utt_index, utterance_dict_sorted[index + 1][0]))

            if start_frame_next - end_frame >= 100:
                end_frame_extend = end_frame + 50
            else:
                end_frame_extend = end_frame + \
                    int((start_frame_next - end_frame) / 2)

        # last uttrerance
        elif index == utt_num - 1:
            if start_frame - end_frame_pre >= 100:
                start_frame_extend = start_frame - 50
            else:
                start_frame_extend = start_frame - \
                    int((start_frame - end_frame_pre) / 2)

            if input_data.shape[0] - end_frame >= 50:
                end_frame_extend = end_frame + 50
            else:
                end_frame_extend = input_data.shape[0]  # last frame

        # middle uttrerances
        else:
            if start_frame - end_frame_pre >= 100:
                start_frame_extend = start_frame - 50
            else:
                start_frame_extend = start_frame - \
                    int((start_frame - end_frame_pre) / 2)

            start_frame_next = int(
                float(utterance_dict_sorted[index + 1][1][0]) * 100 + 0.05)
            if end_frame > start_frame_next:
                print('Warning: utterances are overlapping.')
                print('speaker: %s' % speaker_name)
                print('utt index: %s & %s' %
                      (utt_index, utterance_dict_sorted[index + 1][0]))

            if start_frame_next - end_frame >= 100:
                end_frame_extend = end_frame + 50
            else:
                end_frame_extend = end_frame + \
                    int((start_frame_next - end_frame) / 2)

        input_data_utt = input_data[start_frame_extend:end_frame_extend]
        total_frame_num += (end_frame_extend - start_frame_extend)
        input_data_dict[speaker_name + '_' + utt_index] = input_data_utt

        # update
        end_frame_pre = end_frame

    return input_data_dict
