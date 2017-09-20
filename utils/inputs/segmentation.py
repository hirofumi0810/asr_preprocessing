#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Segment a htk file into each utterance."""

import numpy as np
from struct import unpack

from utils.inputs.wav2feature_python_speech_features import wav2feature as w2f_psf
from utils.inputs.wav2feature_librosa import wav2feature as w2f_librosa


def segment_htk(audio_path, speaker, utterance_dict, is_training,
                sil_duration=0., tool='htk', config=None, mean=None,
                dtype=np.float64):
    """Segment each HTK or WAV file into utterances. Normalization will not be
       conducted here.
    Args:
        audio_path (string): path to a HTK or WAV file
        speaker (string): speaker name
        utterance_dict (dict): dictionary of utterance information
            key (string) => utterance index
            value (list) => [start_frame, end_frame, transcript (, transcript2)]
        sil_duration (float): duration of silence at both ends. Default is 0.
        tool (string): htk or python_speech_features or librosa
        config (dict): a configuration for feature extraction
        mean (np.ndarray):  A mean vector over the file
        dtype (optional): default is np.float64
    Returns:
        input_data_dict (dict):
            key (string) => utt_index
            value (np.ndarray )=> a feature vector of size
                `(frame_num, feature_dim)`
        input_data_utt_sum (np.ndarray): A sum of feature vectors of a speaker
        mean (np.ndarray): A mean vector over the file
        stddev (np.ndarray): A stddev vector over the file
        total_frame_num_file (int): total frame num of the target speaker's utterances
    """
    if tool != 'htk' and config is None:
        raise ValueError('Set config dict.')

    # Read the HTK or WAV file
    if tool == 'htk':
        input_data = read_htk(audio_path)
    elif tool == 'python_speech_features':
        input_data = w2f_psf(audio_path,
                             feature_type=config['feature_type'],
                             feature_dim=config['channels'],
                             use_energy=config['energy'],
                             use_delta1=config['delta'],
                             use_delta2=config['deltadelta'],
                             window=config['window'],
                             slide=config['slide'])

    elif tool == 'librosa':
        input_data = w2f_librosa(audio_path,
                                 feature_type=config['feature_type'],
                                 feature_dim=config['channels'],
                                 use_energy=config['energy'],
                                 use_delta1=config['delta'],
                                 use_delta2=config['deltadelta'],
                                 window=config['window'],
                                 slide=config['slide'])

    feature_dim = input_data.shape[1]

    # Divide into each utterance
    input_data_dict = {}
    total_frame_num_file = 0
    end_frame_pre = 0
    utt_num = len(utterance_dict.keys())
    utt_dict_sorted = sorted(utterance_dict.items(), key=lambda x: x[0])
    input_data_utt_sum = np.zeros((feature_dim,), dtype=dtype)
    stddev = np.zeros((feature_dim,), dtype=dtype)
    for i, (utt_index, utt_info) in enumerate(utt_dict_sorted):
        start_frame, end_frame = utt_info[0], utt_info[1]

        # Check timestamp
        if start_frame > end_frame:
            print(utterance_dict)
            print('Warning: time stamp is reversed.')
            print('speaker index: %s' % speaker)
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
                print('speaker index: %s' % speaker)
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
                print('speaker: %s' % speaker)
                print('utt index: %s & %s' %
                      (str(utt_index), utt_dict_sorted[i + 1][0]))

            if start_frame_next - end_frame >= sil_duration * 2:
                end_frame_extend = end_frame + sil_duration
            else:
                end_frame_extend = end_frame + \
                    int((start_frame_next - end_frame) / 2)

        input_data_utt = input_data[start_frame_extend:end_frame_extend]
        input_data_utt_sum += np.sum(input_data_utt, axis=0)
        total_frame_num_file += (end_frame_extend - start_frame_extend)
        input_data_dict[str(utt_index)] = input_data_utt

        # For computing stddev over the file
        if mean is not None:
            stddev += np.sum(
                np.abs(input_data_utt - mean) ** 2, axis=0)

        # Update
        end_frame_pre = end_frame

    if is_training:
        if mean is not None:
            # Compute stddev over the file
            stddev = np.sqrt(stddev / (total_frame_num_file - 1))
        else:
            # Compute mean over the file
            mean = input_data_utt_sum / total_frame_num_file
            stddev = None
    else:
        mean, stddev = None, None

    return input_data_dict, input_data_utt_sum, mean, stddev, total_frame_num_file


def read_htk(audio_path):
    """Read each HTK file.
    Args:
        audio_path (string): path to a HTK file
    Returns:
        input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
    """
    with open(audio_path, "rb") as fh:
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
