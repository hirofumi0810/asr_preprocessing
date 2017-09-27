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

from utils.util import mkdir_join
from utils.inputs.segmentation import read_htk as read_htk_utt
from utils.inputs.wav2feature_python_speech_features import wav2feature as w2f_psf
from utils.inputs.wav2feature_librosa import wav2feature as w2f_librosa


# TODO: compute male & female statisics

def read_audio(audio_paths, tool, config, normalize, is_training, save_path=None,
               train_global_mean_male=None, train_global_std_male=None,
               train_global_mean_female=None, train_global_std_female=None,
               dtype=np.float64):
    """Read WAV or HTK files.
    Args:
        audio_paths (list): paths to WAV files
        tool (string): the tool to extract features,
            htk or librosa or python_speech_features
        config (dict): a configuration for feature extraction
        normalize (string):
            global => normalize input features by global mean & std over
                      the training set per gender
            speaker => normalize input features by mean & std per speaker
            utterance => normalize input features by mean & std per utterancet
                         data by mean & std per utterance
        is_training (bool, optional): training or not
        save_path (string): path to save npy files
        train_global_mean_male (np.ndarray, optional): global mean of male over
            the training set
        train_global_std_male (np.ndarray, optional): global standard deviation
            of male over the training set
        train_global_mean_female (np.ndarray, optional): global mean of female
            over the training set
        train_global_std_female (np.ndarray, optional): global standard
            deviation of female over the training set
        dtype (optional):
    Returns:
        train_global_mean_male (np.ndarray): global mean of male over the
            training set
        train_global_std_male (np.ndarray): global standard deviation of male
            over the training set
        train_global_mean_female (np.ndarray): global mean of female over the
            training set
        train_global_std_female (np.ndarray): global standard deviation of
            female over the training set
    """
    if not is_training:
        if train_global_mean_male is None or train_global_std_male is None:
            raise ValueError(
                'Set global mean & std computed over the training set.')
    if normalize not in ['global', 'speaker', 'utterance']:
        raise ValueError('normalize is "utterance" or "speaker" or "global".')

    # Read each audio file
    print('===> Reading audio files...')
    wav_paths_male, wav_paths_female = [], []
    input_data_list_male, input_data_list_female = [], []
    total_frame_num_male, total_frame_num_female = 0, 0
    total_frame_num_dict = {}
    speaker_mean_dict, speaker_std_dict = {}, {}
    for audio_path in tqdm(audio_paths):
        speaker = audio_path.split('/')[-2]
        gender = speaker[0]  # f (female) or m (male)
        utt_index = basename(audio_path).split('.')[0]

        if tool == 'htk':
            input_data_utt = read_htk_utt(audio_path)
            # NOTE: audio_path is a htk file path in this case

        elif tool == 'python_speech_features':
            input_data_utt = w2f_psf(audio_path,
                                     feature_type=config['feature_type'],
                                     feature_dim=config['channels'],
                                     use_energy=config['energy'],
                                     use_delta1=config['delta'],
                                     use_delta2=config['deltadelta'],
                                     window=config['window'],
                                     slide=config['slide'])

        elif tool == 'librosa':
            input_data_utt = w2f_librosa(audio_path,
                                         feature_type=config['feature_type'],
                                         feature_dim=config['channels'],
                                         use_energy=config['energy'],
                                         use_delta1=config['delta'],
                                         use_delta2=config['deltadelta'],
                                         window=config['window'],
                                         slide=config['slide'])

        if gender == 'm':
            input_data_list_male.append(input_data_utt)
            wav_paths_male.append(audio_path)
        elif gender == 'f':
            input_data_list_female.append(input_data_utt)
            wav_paths_female.append(audio_path)
        else:
            raise ValueError

        if is_training:
            speaker = audio_path.split('/')[-2]
            gender = speaker[0]
            frame_num_utt, feat_dim = input_data_utt.shape

            if gender == 'm':
                total_frame_num_male += frame_num_utt
            elif gender == 'f':
                total_frame_num_female += frame_num_utt
            else:
                raise ValueError

            if normalize == 'speaker':
                # Initialization
                if speaker not in total_frame_num_dict.keys():
                    total_frame_num_dict[speaker] = 0
                    speaker_mean_dict[speaker] = np.zeros((feat_dim,),
                                                          dtype=dtype)
                    speaker_std_dict[speaker] = np.zeros((feat_dim,),
                                                         dtype=dtype)

                total_frame_num_dict[speaker] += frame_num_utt
                speaker_mean_dict[speaker] += np.sum(input_data_utt, axis=0)
    # NOTE: load all data in advance because TIMIT is a small dataset.
    # TODO: make this pararell

    if is_training:
        # Compute speaker mean
        if normalize == 'speaker':
            for speaker in speaker_mean_dict.keys():
                speaker_mean_dict[speaker] /= total_frame_num_dict[speaker]

        # Compute global mean & std per gender
        print('===> Computing global mean & std over the training set...')
        frame_offset = 0
        feat_dim = input_data_list_male[0].shape[1]
        train_data_male = np.empty((total_frame_num_male, feat_dim))
        train_data_female = np.empty((total_frame_num_female, feat_dim))
        # male
        for input_data_utt, audio_path in zip(tqdm(input_data_list_male),
                                              wav_paths_male):
            speaker = audio_path.split('/')[-2]
            frame_num_utt = input_data_utt.shape[0]
            train_data_male[frame_offset:frame_offset +
                            frame_num_utt] = input_data_utt
            frame_offset += frame_num_utt

            if normalize == 'speaker':
                speaker_std_dict[speaker] += np.sum(
                    np.abs(input_data_utt -
                           speaker_mean_dict[speaker]) ** 2, axis=0)
        # female
        frame_offset = 0
        for input_data_utt, audio_path in zip(tqdm(input_data_list_female),
                                              wav_paths_female):
            speaker = audio_path.split('/')[-2]
            frame_num_utt = input_data_utt.shape[0]
            train_data_female[frame_offset:frame_offset +
                              frame_num_utt] = input_data_utt
            frame_offset += frame_num_utt

            if normalize == 'speaker':
                speaker_std_dict[speaker] += np.sum(
                    np.abs(input_data_utt -
                           speaker_mean_dict[speaker]) ** 2, axis=0)

        # Compute speaker std
        if normalize == 'speaker':
            for speaker in speaker_std_dict.keys():
                speaker_std_dict[speaker] = np.sqrt(
                    speaker_std_dict[speaker] / (total_frame_num_dict[speaker] - 1))

        train_global_mean_male = np.mean(train_data_male, axis=0)
        train_global_std_male = np.std(train_data_male, axis=0)
        train_global_mean_female = np.mean(train_data_female, axis=0)
        train_global_std_female = np.std(train_data_female, axis=0)

        if save_path is not None:
            # Save global mean & std
            np.save(join(save_path, 'train_global_mean_male.npy'),
                    train_global_mean_male)
            np.save(join(save_path, 'train_global_std_male.npy'),
                    train_global_std_male)
            np.save(join(save_path, 'train_global_mean_female.npy'),
                    train_global_mean_female)
            np.save(join(save_path, 'train_global_std_female.npy'),
                    train_global_std_female)

    if save_path is not None:
        # Save input features as npy files
        print('===> Saving input features...')
        frame_num_dict = {}
        for input_data_utt, audio_path in zip(tqdm(input_data_list_male + input_data_list_female),
                                              wav_paths_male + wav_paths_female):
            speaker = audio_path.split('/')[-2]
            utt_index = basename(audio_path).split('.')[0]
            gender = speaker[0]

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
                if gender == 'm':
                    input_data_utt -= train_global_mean_male
                    input_data_utt /= train_global_std_male
                elif gender == 'f':
                    input_data_utt -= train_global_mean_female
                    input_data_utt /= train_global_std_female
                else:
                    raise ValueError

            np.save(mkdir_join(save_path, speaker + '_' +
                               utt_index + '.npy'), input_data_utt)
            frame_num_dict[speaker + '_' + utt_index] = input_data_utt.shape[0]

        # Save a frame number dictionary
        with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
            pickle.dump(frame_num_dict, f)

    return (train_global_mean_male, train_global_std_male,
            train_global_mean_female, train_global_std_female)
