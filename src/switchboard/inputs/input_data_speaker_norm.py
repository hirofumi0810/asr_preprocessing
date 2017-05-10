#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (Switchboard corpus).
   Normalize per speaker.
"""

import os
import pickle
import numpy as np
import functools
from tqdm import tqdm
np.seterr(all='ignore')

from utils.util import mkdir
from utils.inputs.read_htk import read


def read_htk(htk_paths,  speaker_dict, normalize, save_path=None):
    """Read HTK files. This function is used only when training.
    Args:
        htk_paths: list of paths to HTK files
        speaker_dict: dictionary of speakers
            key => speaker index
            value => dictionary of utterance infomation of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
        normalize: if True, normalize by mean & std of each speaker
        save_path: path to save npy files
    """
    # load each HTK file (multiprocessing)
    print('===> Reading HTK files...')
    return_list = []
    for htk_path in htk_paths:
        speaker_index = os.path.basename(htk_path).split('.')[0]
        # e.g. sw04771A => sw4771A (LDC97S62)
        speaker_index = speaker_index.replace('sw0', 'sw')

        return_list.append(
            read_each_htk(htk_path, speaker_index, speaker_dict[speaker_index], normalize))
    input_data_dict_list, train_mean_list, train_std_list,  total_frame_num_list = [], [], [], []
    for return_element in return_list:
        input_data_dict_list.append(return_element[0])
        train_mean_list.append(return_element[1])
        train_std_list.append(return_element[2])
        total_frame_num_list.append(return_element[3])

    # merge all dicts
    print('===> Merge all dictionaries...')
    all_input_data_dict = functools.reduce(lambda first, second: dict(first, **second),
                                           input_data_dict_list)

    if normalize:
        # count total frame num
        total_frame_num = sum(total_frame_num_list)

        # compute global mean & std
        print('===> Computing global mean & std over train data...')
        feature_dim = 123
        train_mean_sum, train_std_sum = np.zeros((feature_dim,)), np.zeros((feature_dim,))
        for train_mean_speaker, train_std_speaker, total_frame_num_speaker in tqdm(zip(train_mean_list,
                                                                                       train_std_list,
                                                                                       total_frame_num_list)):
            train_mean_sum += train_mean_speaker * total_frame_num_speaker
            train_std_sum += train_std_speaker * total_frame_num_speaker
        train_mean_global = train_mean_sum / total_frame_num
        train_std_global = train_std_sum / total_frame_num  # TODO: fix

        if save_path is not None:
            # save global mean & std
            statistic_save_path = '/'.join(save_path.split('/')[:-1])
            np.save(os.path.join(statistic_save_path, 'train_mean.npy'), train_mean_global)
            np.save(os.path.join(statistic_save_path, 'train_std.npy'), train_std_global)

    if save_path is not None:
        # save input data
        print('===> Saving input data...')
        frame_num_dict = {}
        for input_data_save_name, input_data_utt in tqdm(all_input_data_dict.items()):
            speaker_index, utt_index = input_data_save_name.split('_')
            mkdir(os.path.join(save_path, speaker_index))
            input_data_save_path = os.path.join(
                save_path, speaker_index, input_data_save_name + '.npy')

            np.save(input_data_save_path, input_data_utt)
            frame_num_dict[input_data_save_name] = input_data_utt.shape[0]

        # save frame a number dictionary
        frame_num_dict_save_path = '/'.join(save_path.split('/')[:-1])
        with open(os.path.join(frame_num_dict_save_path, 'frame_num.pickle'), 'wb') as f:
            print('===> Saving : frame_num.pickle')
            pickle.dump(frame_num_dict, f)


def read_each_htk(htk_path, speaker_index, utterance_dict, normalize):
    """Read each HTK file.
    Args:
        htk_path: path to a HTK file
        speaker_index: speaker index
        utterance_dict: dictionary of utterance infomation of each speaker
            key => utterance index
            value => [start_frame, end_frame, transcript]
        normalize: if True, normalize by mean & std of each speaker
    Returns:
        input_data_dict:
            key => speaker_index + utterance_index
            value => np.ndarray, (frame_num, feature_dim)
        train_mean: mean per speaker
        train_std: standard deviation per speaker
        total_frame_num: total frame num per speaker
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
            print('Warning: time stamp is reversed.')

        # first utterance
        if index == 0:
            if start_frame >= 50:
                start_frame_extend = start_frame - 50
            else:
                start_frame_extend = 0

            start_frame_next = utterance_dict_sorted[index + 1][1][0]
            if end_frame > start_frame_next:
                print('Warning: utterances are overlapping.')
                print('speaker: %s' % speaker_index)
                print('utt index: %s & %s' % (utt_index, utterance_dict_sorted[index + 1][0]))

            if start_frame_next - end_frame >= 100:
                end_frame_extend = end_frame + 50
            else:
                end_frame_extend = end_frame + int((start_frame_next - end_frame) / 2)

        # last uttrerance
        elif index == utt_num - 1:
            if start_frame - end_frame_pre >= 100:
                start_frame_extend = start_frame - 50
            else:
                start_frame_extend = start_frame - int((start_frame - end_frame_pre) / 2)

            if input_data.shape[0] - end_frame >= 50:
                end_frame_extend = end_frame + 50
            else:
                end_frame_extend = input_data.shape[0]  # last frame

        # middle uttrerances
        else:
            if start_frame - end_frame_pre >= 100:
                start_frame_extend = start_frame - 50
            else:
                start_frame_extend = start_frame - int((start_frame - end_frame_pre) / 2)

            start_frame_next = int(float(utterance_dict_sorted[index + 1][1][0]) * 100 + 0.05)
            if end_frame > start_frame_next:
                print('Warning: utterances are overlapping.')
                print('speaker: %s' % speaker_index)
                print('utt index: %s & %s' % (utt_index, utterance_dict_sorted[index + 1][0]))

            if start_frame_next - end_frame >= 100:
                end_frame_extend = end_frame + 50
            else:
                end_frame_extend = end_frame + int((start_frame_next - end_frame) / 2)

        input_data_utt = input_data[start_frame_extend:end_frame_extend]
        total_frame_num += (end_frame_extend - start_frame_extend)
        input_data_dict[speaker_index + '_' + utt_index] = input_data_utt

        # update
        end_frame_pre = end_frame

    if normalize:
        # compute mean & std per speaker (including silence)
        frame_offset = 0
        feature_dim = input_data.shape[1]
        train_data = np.empty((total_frame_num, feature_dim))
        utterance_dict_sorted = sorted(input_data_dict.items(), key=lambda x: x[0])
        for index, (utt_index, utt_info) in enumerate(utterance_dict_sorted):
            frame_num_utt = input_data_utt.shape[0]
            train_data[frame_offset:frame_offset + frame_num_utt] = input_data_utt
            frame_offset += frame_num_utt
            total_frame_num -= frame_num_utt
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)

        # normalize by mean & std per speaker
        for utt_index, input_data_utt in input_data_dict.items():
            input_data_utt = (input_data_utt - train_mean) / train_std
            input_data_dict[utt_index] = input_data_utt

    else:
        train_mean, train_std = None, None

    return input_data_dict, train_mean, train_std, total_frame_num
