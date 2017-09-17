#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input features (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import pickle
import numpy as np
from tqdm import tqdm

from utils.util import mkdir_join
from utils.inputs.segmentation import segment_htk


def read_audio(audio_paths, speaker_dict, tool, config, normalize, is_training,
               save_path=None,
               train_global_mean_male=None, train_global_mean_female=None,
               train_global_std_male=None, train_global_std_female=None,
               dtype=np.float64):
    """Read HTK or WAV files.
    Args:
        audio_paths (list): paths to HTK or WAV files
        speaker_dict (dict): dictionary of speakers
            key => speaker
            value => dictionary of utterance information of each speaker
                key => utterance index
                value => [start_frame, end_frame, trans_kana, trans_kanji]
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
        dtype (optional): default is np.float64
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
        if train_global_mean_male is None or train_global_mean_female is None or train_global_std_male is None or train_global_std_female is None:
            raise ValueError('Set mean & std computed in the training set.')
    if normalize not in ['global', 'speaker', 'utterance']:
        raise ValueError('normalize is "utterance" or "speaker" or "global".')

    audio_path_list_male, audio_path_list_female = [], []
    total_frame_num_male, total_frame_num_female = 0, 0
    total_frame_num_dict = {}
    speaker_mean_dict = {}

    # NOTE: speaker norm は講演ごとの正規化とする
    # 講演間の話者関係がわからないから

    # Loop 1: Computing global mean and statistics
    if is_training:
        print('===> Reading WAV files...')
        for i, audio_path in enumerate(tqdm(audio_paths)):
            speaker = basename(audio_path).split('.')[0]

            # Divide each WAV or HTK file into utterances
            _, input_data_utt_sum, speaker_mean, _, total_frame_num_speaker = segment_htk(
                audio_path,
                speaker,
                speaker_dict[speaker],
                is_training=True,
                sil_duration=0,
                tool=tool,
                config=config)

            if i == 0:
                # Initialize global statistics
                feature_dim = input_data_utt_sum.shape[0]
                train_global_mean_male = np.zeros((feature_dim,), dtype=dtype)
                train_global_mean_female = np.zeros((feature_dim,), dtype=dtype)
                train_global_std_male = np.zeros((feature_dim,), dtype=dtype)
                train_global_std_female = np.zeros((feature_dim,), dtype=dtype)

            # For computing global mean
            if speaker[3] == 'M':
                audio_path_list_male.append(audio_path)
                train_global_mean_male += input_data_utt_sum
                total_frame_num_male += total_frame_num_speaker
            elif speaker[3] == 'F':
                audio_path_list_female.append(audio_path)
                train_global_mean_female += input_data_utt_sum
                total_frame_num_female += total_frame_num_speaker

            # For computing speaker stddev
            if normalize == 'speaker':
                speaker_mean_dict[speaker] = speaker_mean
                total_frame_num_dict[speaker] = total_frame_num_speaker
                # NOTE: すでに話者平均は計算できている

        print('===> Computing global mean & stddev...')
        # Compute global mean per gender
        train_global_mean_male /= total_frame_num_male
        train_global_mean_female /= total_frame_num_female

        for audio_path in tqdm(audio_paths):
            speaker = basename(audio_path).split('.')[0]

            # Divide each HTK into utterances
            input_data_dict_speaker, _, _, _, _ = segment_htk(
                audio_path,
                speaker,
                speaker_dict[speaker],
                is_training=True,
                sil_duration=0,
                tool=tool,
                config=config)

            # For computing global stddev
            if speaker[3] == 'M':
                for input_data_utt in input_data_dict_speaker.values():
                    train_global_std_male += np.sum(
                        np.abs(input_data_utt - train_global_mean_male) ** 2, axis=0)
            elif speaker[3] == 'F':
                for input_data_utt in input_data_dict_speaker.values():
                    train_global_std_female += np.sum(
                        np.abs(input_data_utt - train_global_mean_female) ** 2, axis=0)

        # Compute global stddev per gender
        train_global_std_male = np.sqrt(train_global_std_male / (total_frame_num_male - 1))
        train_global_std_female = np.sqrt(
            train_global_std_female / (total_frame_num_female - 1))

        if save_path is not None:
            # Save global mean & std per gender
            np.save(join(save_path, 'train_global_mean_male.npy'),
                    train_global_mean_male)
            np.save(join(save_path, 'train_global_mean_female.npy'),
                    train_global_mean_female)
            np.save(join(save_path, 'train_global_std_male.npy'),
                    train_global_std_male)
            np.save(join(save_path, 'train_global_std_female.npy'),
                    train_global_std_female)

    # Loop 2: Normalization and Saving
    print('===> Normalization...')
    frame_num_dict = {}
    for audio_path in tqdm(audio_paths):
        speaker = basename(audio_path).split('.')[0]

        if normalize == 'speaker' and is_training:
            speaker_mean = speaker_mean_dict[speaker]
        else:
            speaker_mean = None

        # Divide each HTK into utterances
        input_data_dict_speaker, _, speaker_mean, speaker_std, _ = segment_htk(
            audio_path,
            speaker,
            speaker_dict[speaker],
            is_training=is_training,
            sil_duration=0,
            tool=tool,
            config=config,
            speaker_mean=speaker_mean)  # for compute speaker sttdev
        # NOTE: input_data_dict_speaker have been not normalized yet

        for utt_index, input_data_utt in input_data_dict_speaker.items():

            if normalize == 'utterance' and is_training:
                # Normalize by mean & std per utterance
                utt_mean = np.mean(input_data_utt, axis=0, dtype=dtype)
                utt_std = np.std(input_data_utt, axis=0, dtype=dtype)
                input_data_utt = (input_data_utt - utt_mean) / utt_std

            elif normalize == 'speaker' and is_training:
                # Normalize by mean & std per speaker
                input_data_utt = (input_data_utt - speaker_mean) / speaker_std

            else:
                # Normalize by mean & std over the training set per gender
                if speaker[3] == 'M':
                    input_data_utt -= train_global_mean_male
                    input_data_utt /= train_global_std_male
                elif speaker[3] == 'F':
                    input_data_utt -= train_global_mean_female
                    input_data_utt /= train_global_std_female

            if save_path is not None:
                # Save input features
                mkdir_join(save_path, speaker)
                input_data_save_path = join(
                    save_path, speaker, speaker + '_' + utt_index + '.npy')
                np.save(input_data_save_path, input_data_utt)
                frame_num_dict[utt_index] = input_data_utt.shape[0]

    if save_path is not None:
        # Save the frame number dictionary
        print('===> Saving : frame_num.pickle')
        with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
            pickle.dump(frame_num_dict, f)

    return (train_global_mean_male, train_global_mean_female,
            train_global_std_male, train_global_std_female)
