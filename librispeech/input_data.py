#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input features (Librispeech corpus)."""

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
               speaker_gender_dict, save_path=None, save_format=None,
               global_mean_male=None, global_mean_female=None,
               global_std_male=None, global_std_female=None,
               dtype=np.float32):
    """Read audio files.
    Args:
        audio_paths (list): paths to HTK or WAV files
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
        is_training (bool): Set True if save as training set
        speaker_gender_dict (dict): A dictionary of speakers' gender information
            key (string) => speaker
            value (string) => F or M
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
            raise ValueError('Set mean & std computed in the training set.')
    if normalize not in ['global', 'speaker', 'utterance', 'no']:
        raise ValueError(
            'normalize must be "utterance" or "speaker" or "global" or "no".')
    if tool not in ['htk', 'python_speech_features', 'librosa']:
        raise TypeError(
            'tool must be "htk" or "python_speech_features"' +
            ' or "librosa".')

    audio_path_dict = {}
    audio_path_list_male, audio_path_list_female = [], []
    total_frame_num_male, total_frame_num_female = 0, 0
    total_frame_num_dict = {}
    speaker_mean_dict, speaker_std_dict = {}, {}

    # Loop 1: Divide all audio paths into speakers
    print('===> Reading audio files...')
    for i, audio_path in enumerate(tqdm(audio_paths)):
        # ex.) audio_path: speaker-book-utt_index.***
        speaker, book, utt_index = basename(
            audio_path).split('.')[0].split('-')
        if speaker not in audio_path_dict.keys():
            audio_path_dict[speaker] = []
        audio_path_dict[speaker].append(audio_path)

        if is_training:
            # Read each audio file
            if tool == 'htk':
                input_utt, sampPeriod, parmKind = read(audio_path)
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

            input_utt_sum = np.sum(input_utt, axis=0)

            if i == 0:
                # Initialize global statistics
                feature_dim = input_utt.shape[1]
                global_mean_male = np.zeros((feature_dim,), dtype=dtype)
                global_mean_female = np.zeros(
                    (feature_dim,), dtype=dtype)
                global_std_male = np.zeros((feature_dim,), dtype=dtype)
                global_std_female = np.zeros((feature_dim,), dtype=dtype)

            # For computing global mean
            if speaker_gender_dict[speaker] == 'M':
                audio_path_list_male.append(input_utt)
                global_mean_male += input_utt_sum
                total_frame_num_male += input_utt.shape[0]
            elif speaker_gender_dict[speaker] == 'F':
                audio_path_list_female.append(input_utt)
                global_mean_female += input_utt_sum
                total_frame_num_female += input_utt.shape[0]
            else:
                raise ValueError('gender is M or F.')

            # For computing speaker mean
            if normalize == 'speaker':
                if speaker not in total_frame_num_dict.keys():
                    total_frame_num_dict[speaker] = 0
                    # Initialize speaker statistics
                    speaker_mean_dict[speaker] = np.zeros(
                        (feature_dim,), dtype=dtype)
                    speaker_std_dict[speaker] = np.zeros(
                        (feature_dim,), dtype=dtype)
                speaker_mean_dict[speaker] += input_utt_sum
                total_frame_num_dict[speaker] += input_utt.shape[0]

    # Loop 2: Computing global mean and sttdev
    if is_training and normalize != 'no':
        print('===> Computing global mean & stddev...')
        # Compute global mean per gender
        global_mean_male /= total_frame_num_male
        global_mean_female /= total_frame_num_female

        for speaker, audio_paths_speaker in tqdm(audio_path_dict.items()):
            if normalize == 'speaker':
                # Compute speaker mean
                speaker_mean_dict[speaker] /= total_frame_num_dict[speaker]

            for audio_path in audio_paths_speaker:
                speaker, book, utt_index = basename(
                    audio_path).split('.')[0].split('-')

                # Read each audio file
                if tool == 'htk':
                    input_utt, sampPeriod, parmKind = read(audio_path)
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

                # For computing global stddev
                if speaker_gender_dict[speaker] == 'M':
                    global_std_male += np.sum(
                        np.abs(input_utt - global_mean_male) ** 2, axis=0)
                elif speaker_gender_dict[speaker] == 'F':
                    global_std_female += np.sum(
                        np.abs(input_utt - global_mean_female) ** 2, axis=0)
                else:
                    raise ValueError('gender is M or F.')

                if normalize == 'speaker':
                    # For computing speaker stddev
                    speaker_std_dict[speaker] += np.sum(
                        np.abs(input_utt - speaker_mean_dict[speaker]) ** 2, axis=0)

            if normalize == 'speaker':
                # Compute speaker stddev
                speaker_std_dict[speaker] = np.sqrt(
                    speaker_std_dict[speaker] / (total_frame_num_dict[speaker] - 1))

        # Compute global stddev per gender
        global_std_male = np.sqrt(
            global_std_male / (total_frame_num_male - 1))
        global_std_female = np.sqrt(
            global_std_female / (total_frame_num_female - 1))

        if save_path is not None:
            # Save global mean & std per gender
            np.save(join(save_path, 'global_mean_male.npy'),
                    global_mean_male)
            np.save(join(save_path, 'global_mean_female.npy'),
                    global_mean_female)
            np.save(join(save_path, 'global_std_male.npy'),
                    global_std_male)
            np.save(join(save_path, 'global_std_female.npy'),
                    global_std_female)

    # Loop 3: Normalization and Saving
    print('===> Normalization...')
    frame_num_dict = {}
    for speaker, audio_paths_speaker in tqdm(audio_path_dict.items()):
        for audio_path in audio_paths_speaker:
            speaker, book, utt_index = basename(
                audio_path).split('.')[0].split('-')

            # Read each audio file
            if tool == 'htk':
                input_utt, sampPeriod, parmKind = read(audio_path)
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

            if normalize == 'no':
                pass
            elif normalize == 'global' or not is_training:
                # Normalize by mean & std over the training set per gender
                if speaker_gender_dict[speaker] == 'M':
                    input_utt -= global_mean_male
                    input_utt /= global_std_male
                elif speaker_gender_dict[speaker] == 'F':
                    input_utt -= global_mean_female
                    input_utt /= global_std_female
                else:
                    raise ValueError('gender is M or F.')
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
                    input_name = basename(audio_path).split('.')[0]
                    input_data_save_path = mkdir_join(
                        save_path, speaker, input_name + '.npy')
                    np.save(input_data_save_path, input_utt)
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

    return (global_mean_male, global_mean_female,
            global_std_male, global_std_female, frame_num_dict)
