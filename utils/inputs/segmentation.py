#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from struct import unpack
from tqdm import tqdm


def segment_htk(htk_path, speaker_name, utterance_dict, normalize,
                sil_duration=50):
    """Segment each HTK file.
    Args:
        htk_path: path to a HTK file
        speaker_name: speaker name
        utterance_dict: dictionary of utterance information of each speaker
            key => utterance index
            value => [start_frame, end_frame, transcript]
        normalize : if speaker, normalize inputs by mean & std per speaker
        sil_duration: duration of silence at both ends
    Returns:
        input_data_dict:
            key => speaker_name + utterance_index
            value => np.ndarray, (frame_num, feature_dim)
        train_mean: mean per speaker
        train_std: standard deviation per speaker
        total_frame_num: total frame num per speaker
    """
    # Load the htk file
    input_data = load_htk(htk_path)

    # Divide into each utterance
    input_data_dict = {}
    total_frame_num = 0
    end_frame_pre = 0
    utt_num = len(utterance_dict.keys())
    utt_dict_sorted = sorted(utterance_dict.items(), key=lambda x: x[0])
    for index, (utt_index, utt_info) in enumerate(utt_dict_sorted):
        start_frame, end_frame = utt_info[0], utt_info[1]

        # Error check
        if start_frame > end_frame:
            print(utterance_dict)
            print('Warning: time stamp is reversed.')
            print('speaker name: %s' % speaker_name)
            print('utterance index: %s & %s' %
                  (str(utt_index), utt_dict_sorted[index + 1][0]))

        # First utterance
        if index == 0:
            if start_frame >= sil_duration:
                start_frame_extend = start_frame - sil_duration
            else:
                start_frame_extend = 0

            start_frame_next = utt_dict_sorted[index + 1][1][0]
            if end_frame > start_frame_next:
                print('Warning: utterances are overlapping.')
                print('speaker name: %s' % speaker_name)
                print('utterance index: %s & %s' %
                      (str(utt_index), utt_dict_sorted[index + 1][0]))

            if start_frame_next - end_frame >= sil_duration * 2:
                end_frame_extend = end_frame + sil_duration
            else:
                end_frame_extend = end_frame + \
                    int((start_frame_next - end_frame) / 2)

        # Last utterance
        elif index == utt_num - 1:
            if start_frame - end_frame_pre >= sil_duration * 2:
                start_frame_extend = start_frame - sil_duration
            else:
                start_frame_extend = start_frame - \
                    int((start_frame - end_frame_pre) / 2)

            if input_data.shape[0] - end_frame >= sil_duration:
                end_frame_extend = end_frame + sil_duration
            else:
                end_frame_extend = input_data.shape[0]  # last frame

        # Middle utterances
        else:
            if start_frame - end_frame_pre >= sil_duration * 2:
                start_frame_extend = start_frame - sil_duration
            else:
                start_frame_extend = start_frame - \
                    int((start_frame - end_frame_pre) / 2)

            start_frame_next = utt_dict_sorted[index + 1][1][0]
            if end_frame > start_frame_next:
                print('Warning: utterances are overlapping.')
                print('speaker: %s' % speaker_name)
                print('utt index: %s & %s' %
                      (str(utt_index), utt_dict_sorted[index + 1][0]))

            if start_frame_next - end_frame >= sil_duration * 2:
                end_frame_extend = end_frame + sil_duration
            else:
                end_frame_extend = end_frame + \
                    int((start_frame_next - end_frame) / 2)

        input_data_utt = input_data[start_frame_extend:end_frame_extend]
        total_frame_num += (end_frame_extend - start_frame_extend)
        input_data_dict[speaker_name + '_' + str(utt_index)] = input_data_utt

        # Update
        end_frame_pre = end_frame

    # Compute mean & std per speaker (including silence)
    frame_offset = 0
    feature_dim = input_data.shape[1]
    train_data = np.empty((total_frame_num, feature_dim), dtype=np.float64)
    for i, input_data_utt in enumerate(input_data_dict.values()):
        frame_num_utt = input_data_utt.shape[0]
        train_data[frame_offset:frame_offset +
                   frame_num_utt] = input_data_utt
        frame_offset += frame_num_utt
    train_mean = np.mean(train_data, axis=0, dtype=np.float64)
    train_std = np.std(train_data, axis=0, dtype=np.float64)

    if normalize == 'speaker':
        # Normalize by mean & std per speaker
        for key, input_data_utt in input_data_dict.items():
            input_data_utt = (input_data_utt - train_mean) / train_std
            input_data_dict[key] = input_data_utt

    return input_data_dict, train_mean, train_std, total_frame_num


def load_htk(htk_path):
    """Load each HTK file.
    Args:
        htk_path: path to a HTK file
    Returns:
        input_data: np.ndarray, (frame_num, feature_dim)
    """
    with open(htk_path, "rb") as fh:
        spam = fh.read(12)
        frame_num, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
        # print(frame_num)  # frame num
        # print(sampPeriod)  # 10ms
        # print(sampSize)  # feature dim * 4 (byte)
        # print(parmKind)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        input_data = np.fromfile(fh, 'f')
        # input_data = input_data.reshape(int(len(input_data) / veclen),
        # veclen)
        input_data = input_data.reshape(-1, veclen)
        input_data.byteswap(True)

    return input_data


def global_mean(mean_list, total_frame_num_list):
    """Calculate global mean of a number of means.
    Args:
        mean_list: list of mean of each speaker
        total_frame_num_list: list of frame num of each mean
    Returns:
        global_mean_value: global mean value over overall data
    """
    feature_dim = mean_list[0].shape[0]
    global_mean_value = np.empty((feature_dim,), dtype=np.float64)
    total_frame_num = np.sum(total_frame_num_list)
    for i in tqdm(range(len(mean_list))):
        total_frame_num_speaker = total_frame_num_list[i]
        global_mean_value += mean_list[i] * float(total_frame_num_speaker)
    global_mean_value = global_mean_value / float(total_frame_num)
    return global_mean_value


def global_std(input_data_dict_list, total_frame_num_list, global_mean_value):
    """Calculate global standard deviation of a number of stds.
    Args:
        input_data_dict_list: list of dictionaries of input data
        total_frame_num_list: list of frame num of each std
        global_mean_value: global mean value over overall data
    Returns:
        global_std_value: global std value over overall data
    """
    feature_dim = global_mean_value.shape[0]
    global_std_value = np.empty((feature_dim,), dtype=np.float64)
    total_frame_num = np.sum(total_frame_num_list)
    for i, input_data_dict in enumerate(tqdm(input_data_dict_list)):
        # Compute square values between features & global mean per speaker
        frame_offset = 0
        total_frame_num_speaker = total_frame_num_list[i]
        input_data_speaker = np.empty((total_frame_num_speaker, feature_dim))
        for input_data_utt in input_data_dict.values():
            frame_num_utt = input_data_utt.shape[0]
            input_data_speaker[frame_offset:frame_offset +
                               frame_num_utt] = input_data_utt
            frame_offset += frame_num_utt
        global_std_value += sqrt_sum(input_data_speaker, global_mean_value)
    global_std_value = np.power(global_std_value / float(total_frame_num), 0.5)
    return global_std_value


def sqrt_sum(input_data, global_mean_value):
    """Calculate square sum, sigma(each_input - global_mean_value)^2.
    Args:
        input_data: input data of each speaker
        global_mean_value: global mean value over overall data
    Returns:
        value_sqrt_sum: square sum
    """
    feature_dim = input_data[0].shape[0]
    value_sqrt_sum = np.zeros((feature_dim,))
    for input_vec in input_data:
        value_sqrt_sum += np.power((input_vec -
                                    global_mean_value), 2, dtype=np.float64)
    return value_sqrt_sum
