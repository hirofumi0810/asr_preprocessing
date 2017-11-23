#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input features (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import numpy as np
import pickle
from tqdm import tqdm

from utils.util import mkdir_join
from utils.inputs.htk import read, write
from utils.inputs.wav2feature_python_speech_features import wav2feature as w2f_psf
from utils.inputs.wav2feature_librosa import wav2feature as w2f_librosa


def read_audio(audio_paths, tool, config, normalize, is_training,
               save_path=None, save_format=None,
               global_mean_male=None, global_std_male=None,
               global_mean_female=None, global_std_female=None,
               dtype=np.float32):
    """Read audio files.
    Args:
        audio_paths (list): paths to audio files
        tool (string): the tool to extract features,
            htk or librosa or python_speech_features
        config (dict): a configuration for feature extraction
        normalize (string):
            no => normalization will be not conducted
            global => normalize input features by global mean & std over
                      the training set per gender
            speaker => normalize input features by mean & std per speaker
            utterance => normalize input features by mean & std per utterancet
                         data by mean & std per utterance
        is_training (bool, optional):  Set True when proccessing the training set
        save_path (string): path to save npy files
        save_format (string, optional): numpy as htk
        global_mean_male (np.ndarray, optional): global mean of male over
            the training set
        global_std_male (np.ndarray, optional): global standard deviation
            of male over the training set
        global_mean_female (np.ndarray, optional): global mean of female
            over the training set
        global_std_female (np.ndarray, optional): global standard
            deviation of female over the training set
        dtype (optional): the type of data, default is np.float32
    Returns:
        global_mean_male (np.ndarray): global mean of male over the
            training set
        global_std_male (np.ndarray): global standard deviation of male
            over the training set
        global_mean_female (np.ndarray): global mean of female over the
            training set
        global_std_female (np.ndarray): global standard deviation of
            female over the training set
        frame_num_dict (dict):
            key => utterance name
            value => the number of frames
    """
    if not is_training:
        if global_mean_male is None or global_std_male is None:
            raise ValueError(
                'Set global mean & std computed over the training set.')
    if normalize not in ['global', 'speaker', 'utterance', 'no']:
        raise ValueError(
            'normalize must be "utterance" or "speaker" or "global" or "no".')

    # Read each audio file
    print('=====> Reading audio files...')
    audio_paths_male, audio_paths_female = [], []
    input_data_list_male, input_data_list_female = [], []
    total_frame_num_male, total_frame_num_female = 0, 0
    total_frame_num_dict = {}
    speaker_mean_dict, speaker_std_dict = {}, {}
    for audio_path in tqdm(audio_paths):
        speaker = audio_path.split('/')[-2]
        gender = speaker[0]  # f (female) or m (male)
        utt_index = basename(audio_path).split('.')[0]

        if tool == 'htk':
            input_utt, sampPeriod, parmKind = read(audio_path)
            # NOTE: audio_path is a htk file path in this case
        elif tool == 'python_speech_features':
            input_utt = w2f_psf(
                audio_path,
                feature_type=config['feature_type'],
                feature_dim=config['channels'],
                use_energy=config['energy'],
                use_delta1=config['delta'],
                use_delta2=config['deltadelta'],
                window=config['window'],
                slide=config['slide'])
        elif tool == 'librosa':
            input_utt = w2f_librosa(
                audio_path,
                feature_type=config['feature_type'],
                feature_dim=config['channels'],
                use_energy=config['energy'],
                use_delta1=config['delta'],
                use_delta2=config['deltadelta'],
                window=config['window'],
                slide=config['slide'])

        # for debug
        # print(input_utt.shape)

        if gender == 'm':
            input_data_list_male.append(input_utt)
            audio_paths_male.append(audio_path)
        elif gender == 'f':
            input_data_list_female.append(input_utt)
            audio_paths_female.append(audio_path)
        else:
            raise ValueError('gender is m or f.')

        if is_training:
            speaker = audio_path.split('/')[-2]
            gender = speaker[0]
            frame_num_utt, feat_dim = input_utt.shape

            if gender == 'm':
                total_frame_num_male += frame_num_utt
            elif gender == 'f':
                total_frame_num_female += frame_num_utt
            else:
                raise ValueError('gender is m or f.')

            if normalize == 'speaker':
                # Initialization
                if speaker not in total_frame_num_dict.keys():
                    total_frame_num_dict[speaker] = 0
                    speaker_mean_dict[speaker] = np.zeros((feat_dim,),
                                                          dtype=dtype)
                    speaker_std_dict[speaker] = np.zeros((feat_dim,),
                                                         dtype=dtype)

                total_frame_num_dict[speaker] += frame_num_utt
                speaker_mean_dict[speaker] += np.sum(input_utt, axis=0)
    # NOTE: Load all data in advance because TIMIT is a small dataset.

    if is_training and normalize != 'no':
        # Compute speaker mean
        if normalize == 'speaker':
            for speaker in speaker_mean_dict.keys():
                speaker_mean_dict[speaker] /= total_frame_num_dict[speaker]

        # Compute global mean & std per gender
        print('=====> Computing global mean & std over the training set...')
        frame_offset = 0
        feat_dim = input_data_list_male[0].shape[1]
        train_data_male = np.empty((total_frame_num_male, feat_dim))
        train_data_female = np.empty((total_frame_num_female, feat_dim))
        # male
        for input_utt, audio_path in zip(tqdm(input_data_list_male),
                                         audio_paths_male):
            speaker = audio_path.split('/')[-2]
            frame_num_utt = input_utt.shape[0]
            train_data_male[frame_offset:frame_offset +
                            frame_num_utt] = input_utt
            frame_offset += frame_num_utt

            if normalize == 'speaker':
                speaker_std_dict[speaker] += np.sum(
                    np.abs(input_utt -
                           speaker_mean_dict[speaker]) ** 2, axis=0)
        # female
        frame_offset = 0
        for input_utt, audio_path in zip(tqdm(input_data_list_female),
                                         audio_paths_female):
            speaker = audio_path.split('/')[-2]
            frame_num_utt = input_utt.shape[0]
            train_data_female[frame_offset:frame_offset +
                              frame_num_utt] = input_utt
            frame_offset += frame_num_utt

            if normalize == 'speaker':
                speaker_std_dict[speaker] += np.sum(
                    np.abs(input_utt -
                           speaker_mean_dict[speaker]) ** 2, axis=0)

        # Compute speaker std
        if normalize == 'speaker':
            for speaker in speaker_std_dict.keys():
                speaker_std_dict[speaker] = np.sqrt(
                    speaker_std_dict[speaker] / (total_frame_num_dict[speaker] - 1))

        global_mean_male = np.mean(train_data_male, axis=0)
        global_std_male = np.std(train_data_male, axis=0)
        global_mean_female = np.mean(train_data_female, axis=0)
        global_std_female = np.std(train_data_female, axis=0)

        if save_path is not None:
            # Save global mean & std
            np.save(join(save_path, 'global_mean_male.npy'),
                    global_mean_male)
            np.save(join(save_path, 'global_std_male.npy'),
                    global_std_male)
            np.save(join(save_path, 'global_mean_female.npy'),
                    global_mean_female)
            np.save(join(save_path, 'global_std_female.npy'),
                    global_std_female)

    # Save input features as npy files
    print('=====> Normalization...')
    frame_num_dict = {}
    for input_utt, audio_path in zip(tqdm(input_data_list_male + input_data_list_female),
                                     audio_paths_male + audio_paths_female):
        speaker = audio_path.split('/')[-2]
        utt_index = basename(audio_path).split('.')[0]
        gender = speaker[0]

        if normalize == 'no':
            pass
        elif normalize == 'global' or not is_training:
            # Normalize by global mean & std over the training set
            if gender == 'm':
                input_utt -= global_mean_male
                input_utt /= global_std_male
            elif gender == 'f':
                input_utt -= global_mean_female
                input_utt /= global_std_female
            else:
                raise ValueError('gender is m or f.')
        elif normalize == 'speaker':
            # Normalize by mean & std per speaker
            input_utt -= speaker_mean_dict[speaker]
            input_utt /= speaker_std_dict[speaker]
        elif normalize == 'utterance':
            # Normalize by mean & std per utterance
            utt_mean = np.mean(input_utt, axis=0, dtype=dtype)
            utt_std = np.std(input_utt, axis=0, dtype=dtype)
            input_utt = (input_utt - utt_mean) / utt_std
        else:
            raise ValueError

        frame_num_dict[speaker + '_' + utt_index] = input_utt.shape[0]

        if save_path is not None:
            # Save input features
            if save_format == 'numpy':
                np.save(mkdir_join(save_path, speaker, speaker + '_' +
                                   utt_index + '.npy'), input_utt)
            elif save_format == 'htk':
                write(input_utt,
                      htk_path=mkdir_join(
                          save_path, speaker, speaker + '_' + utt_index + '.htk'),
                      sampPeriod=sampPeriod,
                      parmKind=parmKind)
            else:
                raise ValueError('save_format is numpy or htk.')

    if save_path is not None:
        # Save the frame number dictionary
        with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
            pickle.dump(frame_num_dict, f)

    return (global_mean_male, global_std_male,
            global_mean_female, global_std_female, frame_num_dict)
