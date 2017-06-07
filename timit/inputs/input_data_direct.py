#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data for CTC network (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import os
import pickle
import numpy as np
import audioread
import multiprocessing as mp
from python_speech_features import fbank, logfbank, hz2mel
from utils import mkdir


def read_wav(input_paths, save_path, is_train=False, train_mean=None, train_std=None):

    if (not is_train) and (train_mean == None or train_std == None):
        raise ValueError('Error: Set mean & std!')

    save_path = mkdir(save_path)

    frame_num_dict = {}

    #######################################################
    # compute mean & standard deviation over train dataset
    #######################################################
    if is_train:
        print('computing mean & std over train dataset...')
        # multiprocessing
        p = mp.Pool(mp.cpu_count() - 1)
        args = [(input_path, save_path, False, train_mean, train_std, 0)
                for input_path in input_paths]
        result_tuple = p.map(read_each_wav, args)

        # compute max frame num
        all_frame_num = 0
        max_frame_num = 0
        for r in result_tuple:
            feature, file_name = r
            frame_num = feature.shape[0]
            all_frame_num += frame_num
            frame_num_dict[file_name] = frame_num
            if frame_num > max_frame_num:
                max_frame_num = frame_num

        start_frame = 0
        train_data = np.empty((all_frame_num, 123))
        for r in result_tuple:
            feature = r[0]
            frame_num = feature.shape[0]
            train_data[start_frame:start_frame + frame_num] = feature
            start_frame += frame_num

        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        del train_data
        gc.collect()

    else:

        print('computing max_frame_num over test dataset...')
        # multiprocessing
        p = mp.Pool(mp.cpu_count() - 1)
        args = [(input_path, save_path, False, train_mean, train_std, 0)
                for input_path in input_paths]
        result_tuple = p.map(read_each_wav, args)

        # compute max frame num
        max_frame_num = 0
        for r in result_tuple:
            feature, file_name = r
            frame_num = feature.shape[0]
            frame_num_dict[file_name] = frame_num
            if frame_num > max_frame_num:
                max_frame_num = frame_num
        print('complete!')

    print(max_frame_num)
    #######################################################
    # save dataset
    #######################################################
    # multiprocessing
    p = mp.Pool(mp.cpu_count() - 1)
    args = [(input_path, save_path, True, train_mean, train_std, max_frame_num)
            for input_path in input_paths]
    p.map(read_each_wav, args)

    with open(os.path.join(save_path, 'frame_num.pickle'), 'wb') as f:
        print('saving : frame_num.pickle')
        pickle.dump(frame_num_dict, f)

    return train_mean, train_std


def read_each_wav(args):
    input_path, save_path, is_save, train_mean, train_std, max_frame_num = args

    # if is_save:
    #     print('Processing: %s' % input_path)
    speaker_name = input_path.split('/')[-2]
    file_name = input_path.split('/')[-1].split('.')[0]
    save_file_name = speaker_name + '_' + file_name + '.npy'

    # read wav
    with audioread.audio_open(input_path) as f:
        # print("ch: %d, fs: %d, duration [s]: %.1f" % (f.channels, f.samplerate, f.duration))
        wav_barray = bytearray()
        for buf in f:
            wav_barray.extend(buf)

         # always read as 16bit
        wav_array = np.frombuffer(wav_barray, dtype=np.int16)

        # convert from short to float
        # wav_float = pcm2float(wav_array)

        # print(wav_float.shape)
        # plt.plot(wav_float)
        # plt.show()

        # convert to log mel filterbank & log energy
        filterbank, energy = fbank(wav_array, nfilt=40)
        logfilterbank = np.log(filterbank)
        logenergy = np.log(energy)
        logmelfilterbank = hz2mel(logfilterbank)
        features = np.c_[logmelfilterbank, logenergy]
        delta1 = delta(features, N=2)
        delta2 = delta(delta1, N=2)
        features = np.c_[features, delta1, delta2]
        # print(features.shape)

        # save as npy file
        if is_save:
            # normalization
            features = (features - train_mean) / train_std

            # padding
            pad_frame = max_frame_num - features.shape[0]
            features = np.pad(features, ((0, pad_frame), (0, 0)),
                              'constant', constant_values=0)
            np.save(os.path.join(save_path, save_file_name), features)

    return features, save_file_name.split('.')[0]


def pcm2float(short_ndary):
    """Convert from short to float."""
    float_ndary = np.array(short_ndary, dtype=np.float64)
    return float_ndary
    # return np.where(float_ndary > 0.0, float_ndary / 32767.0, float_ndary /
    # 32768.0)


def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    NUMFRAMES = len(feat)
    feat = np.concatenate(([feat[0] for i in range(N)],
                           feat, [feat[-1] for i in range(N)]))
    denom = sum([2 * i * i for i in range(1, N + 1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(np.sum([n * feat[N + j + n]
                             for n in range(-1 * N, N + 1)], axis=0) / denom)
    return dfeat
