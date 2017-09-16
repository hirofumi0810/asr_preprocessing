#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input features (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import pickle
import numpy as np
from tqdm import tqdm

from utils.inputs.segmentation import read_htk as read_htk_utt
# from utils.inputs.python_speech_features.wav2feature import wav2feature as w2f_psf
# from utils.inputs.librosa.wav2feature import wav2feature as w2f_librosa


def read_wav(wav_paths, tool, config, normalize, is_training, save_path=None,
             train_global_mean=None, train_global_std=None, dtype=np.float64):
    """Read wav files.
    Args:
        wav_paths (list): list of wav paths
        tool (string): the tool to extract features,
            htk or librosa or python_speech_features
        config (dict): a configuration for feature extraction
        save_path (string): path to save npy files
        normalize (string):
            global => normalize input features by global mean & std over
                      the training set per gender
            speaker => normalize input features by mean & std per speaker
            utterance => normalize input features by mean & std per utterancet data
                         by mean & std per utterance
        is_training (bool, optional): training or not
        train_global_mean (np.ndarray, optional): mean over the training set
        train_global_std (np.ndarray, optional): standard deviation over the
            training set
    Returns:
        train_global_mean (np.ndarray): global mean over the training set
        train_global_std (np.ndarray): global standard deviation over the
            training set
    """
    if not is_training:
        if train_global_mean is None or train_global_std is None:
            raise ValueError('Set global mean & std computed in the training set.')
    if normalize not in ['global', 'speaker', 'utterance']:
        raise ValueError(
            'normalize is "utterance" or "speaker" or "global".')

    # Read each wav file
    print('===> Reading wav files...')
    input_data_list = []
    total_frame_num = 0
    total_frame_num_dict = {}
    speaker_mean_dict, speaker_std_dict = {}, {}
    for wav_path in tqdm(wav_paths):
        if tool == 'htk':
            input_data_utt = read_htk_utt(wav_path)
            input_data_list.append(input_data_utt)
            # NOTE: wav_path is a htk file path in this case
        elif tool == 'librosa':
            raise NotImplementedError
        elif tool == 'python_speech_features':
            raise NotImplementedError
            # input_data_list.append(w2f_psf(wav_path, config))

        if is_training:
            speaker = basename(wav_path).split('_')[0]
            frame_num_utt, feat_dim = input_data_utt.shape
            total_frame_num += frame_num_utt
            if normalize == 'speaker':
                # Initialization
                if speaker not in total_frame_num_dict.keys():
                    total_frame_num_dict[speaker] = 0
                    speaker_mean_dict[speaker] = np.zeros((feat_dim,), dtype=dtype)
                    speaker_std_dict[speaker] = np.zeros((feat_dim,), dtype=dtype)

                total_frame_num_dict[speaker] += frame_num_utt
                speaker_mean_dict[speaker] += np.sum(input_data_utt, axis=0)
    # NOTE: load all data in advance because TIMIT is a small dataset.

    if is_training:
        # Compute speaker mean
        if normalize == 'speaker':
            for speaker in speaker_mean_dict.keys():
                speaker_mean_dict[speaker] /= total_frame_num_dict[speaker]

        # Compute global mean & std
        print('===> Computing global mean & std over the training set...')
        frame_offset = 0
        feat_dim = input_data_list[0].shape[1]
        train_data = np.empty((total_frame_num, feat_dim))
        for input_data_utt, wav_path in zip(tqdm(input_data_list), wav_paths):
            speaker = basename(wav_path).split('_')[0]
            frame_num_utt = input_data_utt.shape[0]
            train_data[frame_offset:frame_offset + frame_num_utt] = input_data_utt
            frame_offset += frame_num_utt

            if normalize == 'speaker':
                speaker_std_dict[speaker] += np.sum(np.abs(input_data_utt -
                                                           speaker_mean_dict[speaker]) ** 2, axis=0)

        # Compute speaker std
        if normalize == 'speaker':
            for speaker in speaker_std_dict.keys():
                speaker_std_dict[speaker] = np.sqrt(
                    speaker_std_dict[speaker] / (total_frame_num_dict[speaker] - 1))

        train_global_mean = np.mean(train_data, axis=0)
        train_global_std = np.std(train_data, axis=0)

        if save_path is not None:
            # Save global mean & std
            np.save(join(save_path, 'train_global_mean.npy'), train_global_mean)
            np.save(join(save_path, 'train_global_std.npy'), train_global_std)

    if save_path is not None:
        # Save input features as npy files
        print('===> Saving input features...')
        frame_num_dict = {}
        for input_data_utt, wav_path in zip(tqdm(input_data_list), wav_paths):
            speaker = basename(wav_path).split('_')[0]
            input_data_save_name = basename(wav_path).split('.')[0] + '.npy'
            input_data_save_path = join(save_path, input_data_save_name)

            # Normalize by global mean & std over the training set
            if normalize == 'speaker' and is_training:
                input_data_utt -= speaker_mean_dict[speaker]
                input_data_utt /= speaker_std_dict[speaker]
            elif normalize == 'utterance' and is_training:
                mean_utt = np.mean(input_data_utt, axis=0)
                std_utt = np.std(input_data_utt, axis=0)
                input_data_utt -= mean_utt
                input_data_utt /= std_utt
            else:
                input_data_utt -= train_global_mean
                input_data_utt /= train_global_std

            np.save(input_data_save_path, input_data_utt)
            frame_num_dict[input_data_save_name] = input_data_utt.shape[0]

        # Save a frame number dictionary
        with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
            print('===> Saving : frame_num.pickle')
            pickle.dump(frame_num_dict, f)

    return train_global_mean, train_global_std
