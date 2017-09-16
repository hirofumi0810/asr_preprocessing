#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Segment a htk file into each utterance."""

import numpy as np
from struct import unpack


def segment_htk(htk_path, speaker_index, utterance_dict, normalize,
                is_training, sil_duration=0., speaker_mean=None):
    """Segment each HTK file into utterances.
    Args:
        htk_path (string): path to a HTK file
        speaker_index (int): speaker index
        utterance_dict (dict): dictionary of utterance information of each speaker
            key (string) => utterance index
            value (list) => [start_frame, end_frame, transcript (, transcript2)]
        normalize (string):
            global    => normalize input data by global mean & std over
                         the training set per gender
            speaker   => normalize input data by mean & std per speaker
            utterance => normalize input data by mean & std per utterance
        is_training (bool): training or not
        sil_duration (float): duration of silence at both ends. Default is 0.
        speaker_mean (np.ndarray): mean of the target speaker
    Returns:
        input_data_dict (dict):
            key (string) => speaker_index + utterance_index
            value (np.ndarray )=> A tensor of size (frame_num, feature_dim)
        speaker_mean (np.ndarray): A mean vector of a speaker in the training set
        total_frame_num_speaker (int): total frame num of the target speaker's utterances
    """
    # Read the htk file
    input_data = read_htk(htk_path)
    feature_dim = input_data.shape[1]

    # Divide into each utterance
    input_data_dict = {}
    total_frame_num_speaker = 0
    end_frame_pre = 0
    utt_num = len(utterance_dict.keys())
    utt_dict_sorted = sorted(utterance_dict.items(), key=lambda x: x[0])
    input_data_sum = np.zeros((feature_dim,), dtype=np.float64)
    speaker_std = np.zeros((feature_dim,), dtype=np.float64)
    for i, (utt_index, utt_info) in enumerate(utt_dict_sorted):
        start_frame, end_frame = utt_info[0], utt_info[1]

        # Check timestamp
        if start_frame > end_frame:
            print(utterance_dict)
            print('Warning: time stamp is reversed.')
            print('speaker index: %s' % speaker_index)
            print('utterance index: %s & %s' %
                  (str(utt_index), utt_dict_sorted[i + 1][0]))

        # Check the first utterance
        if i == 0:
            if start_frame >= sil_duration:
                start_frame_extend = start_frame - sil_duration
            else:
                start_frame_extend = 0

            start_frame_next = utt_dict_sorted[i + 1][1][0]
            if end_frame > start_frame_next:
                print('Warning: utterances are overlapping.')
                print('speaker index: %s' % speaker_index)
                print('utterance index: %s & %s' %
                      (str(utt_index), utt_dict_sorted[i + 1][0]))

            if start_frame_next - end_frame >= sil_duration * 2:
                end_frame_extend = end_frame + sil_duration
            else:
                end_frame_extend = end_frame + \
                    int((start_frame_next - end_frame) / 2)

        # Check the last utterance
        elif i == utt_num - 1:
            if start_frame - end_frame_pre >= sil_duration * 2:
                start_frame_extend = start_frame - sil_duration
            else:
                start_frame_extend = start_frame - \
                    int((start_frame - end_frame_pre) / 2)

            if input_data.shape[0] - end_frame >= sil_duration:
                end_frame_extend = end_frame + sil_duration
            else:
                end_frame_extend = input_data.shape[0]  # last frame

        # Check other utterances
        else:
            if start_frame - end_frame_pre >= sil_duration * 2:
                start_frame_extend = start_frame - sil_duration
            else:
                start_frame_extend = start_frame - \
                    int((start_frame - end_frame_pre) / 2)

            start_frame_next = utt_dict_sorted[i + 1][1][0]
            if end_frame > start_frame_next:
                print('Warning: utterances are overlapping.')
                print('speaker: %s' % speaker_index)
                print('utt index: %s & %s' %
                      (str(utt_index), utt_dict_sorted[i + 1][0]))

            if start_frame_next - end_frame >= sil_duration * 2:
                end_frame_extend = end_frame + sil_duration
            else:
                end_frame_extend = end_frame + \
                    int((start_frame_next - end_frame) / 2)

        input_data_utt = input_data[start_frame_extend:end_frame_extend]
        input_data_sum += np.sum(input_data_utt, axis=0)
        total_frame_num_speaker += (end_frame_extend - start_frame_extend)
        input_data_dict[speaker_index + '_' + str(utt_index)] = input_data_utt

        # For computing speaker stddev
        if speaker_mean is not None:
            speaker_std += np.sum(
                np.abs(input_data_utt - speaker_mean) ** 2, axis=0)

        # Update
        end_frame_pre = end_frame

    if speaker_mean is not None:
        # Compute speaker stddev
        speaker_std /= total_frame_num_speaker
    else:
        # Compute speaker mean
        speaker_mean = input_data_sum / total_frame_num_speaker

    return input_data_dict, input_data_sum, speaker_mean, speaker_std, total_frame_num_speaker


def read_htk(htk_path):
    """Read each HTK file.
    Args:
        htk_path (string): path to a HTK file
    Returns:
        input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
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
