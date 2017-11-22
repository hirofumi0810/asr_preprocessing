#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import numpy as np
import pickle
from tqdm import tqdm

from utils.util import mkdir_join
from utils.inputs.segmentation import segment
from utils.inputs.htk import read, write


def read_audio(audio_paths, speaker_dict, tool, config, normalize, is_training,
               save_path=None, save_format=None, global_mean=None, global_std=None,
               dtype=np.float32):
    """Read HTK or WAV files.
    Args:
        audio_paths (list): paths to HTK or WAV files
        speaker_dict (dict): A dictionary of speakers' gender information
            key (string) => speaker
            value (dict) => dictionary of utterance information of each speaker
                key (string) => utterance index
                value (list) => [start_frame, end_frame, transcript]
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
        is_training (bool): training or not
        save_path (string): path to save npy files
        save_format (string, optional): numpy as htk
        global_mean (np.ndarray, optional): global mean over the training set
        global_std (np.ndarray, optional): global standard deviation over the
            training set
        dtype (optional): the type of data, default is np.float32
    Returns:
        global_mean (np.ndarray): global mean over the training set
        global_std (np.ndarray): global standard deviation over the
            training set
        frame_num_dict (dict):
            key => utterance name
            value => the number of frames
    """
    if not is_training:
        if global_mean is None or global_std is None:
            raise ValueError('Set mean & std computed in the training set.')
    if normalize not in ['global', 'speaker', 'utterance', 'no']:
        raise ValueError(
            'normalize must be "utterance" or "speaker" or "global" or "no".')

    total_frame_num = 0
    total_frame_num_dict = {}
    speaker_mean_dict = {}

    # Loop 1: Computing global mean and statistics
    if is_training and normalize != 'no':
        print('===> Reading audio files...')
        for i, audio_path in enumerate(tqdm(audio_paths)):
            speaker = basename(audio_path).split('.')[0]

            # Fix speaker name
            speaker = speaker.replace('sw0', 'sw')
            # ex.) sw04771-A => sw4771-A (LDC97S62)
            speaker = speaker.replace('sw_', 'sw')
            # ex.) sw_4771-A => sw4771-A (eval2000, swbd)
            speaker = speaker.replace('en_', 'en')
            # ex.) en_4156-A => en4156-A (eval2000, ch)

            # Divide each audio file into utterances
            _, input_utt_sum, speaker_mean, _, total_frame_num_speaker = segment(
                audio_path,
                speaker,
                speaker_dict[speaker],
                is_training=True,
                sil_duration=0,
                tool=tool,
                config=config)

            if i == 0:
                # Initialize global statistics
                feature_dim = input_utt_sum.shape[0]
                global_mean = np.zeros((feature_dim,), dtype=dtype)
                global_std = np.zeros((feature_dim,), dtype=dtype)

            global_mean += input_utt_sum
            total_frame_num += total_frame_num_speaker

            # For computing speaker stddev
            if normalize == 'speaker':
                speaker_mean_dict[speaker] = speaker_mean
                total_frame_num_dict[speaker] = total_frame_num_speaker
                # NOTE: speaker mean is already computed

        print('===> Computing global mean & stddev...')
        # Compute global mean
        global_mean /= total_frame_num

        for audio_path in tqdm(audio_paths):
            speaker = basename(audio_path).split('.')[0]

            # Normalize speaker name
            speaker = speaker.replace('sw0', 'sw')
            speaker = speaker.replace('sw_', 'sw')
            speaker = speaker.replace('en_', 'en')

            # Divide each audio into utterances
            input_data_dict_speaker, _, _, _, _ = segment(
                audio_path,
                speaker,
                speaker_dict[speaker],
                is_training=True,
                sil_duration=0,
                tool=tool,
                config=config)

            # For computing global stddev
            for input_utt in input_data_dict_speaker.values():
                global_std += np.sum(
                    np.abs(input_utt - global_mean) ** 2, axis=0)

        # Compute global stddev
        global_std = np.sqrt(global_std / (total_frame_num - 1))

        if save_path is not None:
            # Save global mean & std per gender
            np.save(join(save_path, 'global_mean.npy'), global_mean)
            np.save(join(save_path, 'global_std.npy'), global_std)

    # Loop 2: Normalization and Saving
    print('===> Normalization...')
    frame_num_dict = {}
    sampPeriod, parmKind = None, None
    for audio_path in tqdm(audio_paths):
        speaker = basename(audio_path).split('.')[0]

        # Normalize speaker name
        speaker = speaker.replace('sw0', 'sw')
        speaker = speaker.replace('sw_', 'sw')
        speaker = speaker.replace('en_', 'en')

        if normalize == 'speaker' and is_training:
            speaker_mean = speaker_mean_dict[speaker]
        else:
            speaker_mean = None

        # Divide each audio into utterances
        input_data_dict_speaker, _, speaker_mean, speaker_std, _ = segment(
            audio_path,
            speaker,
            speaker_dict[speaker],
            is_training=is_training,
            sil_duration=0,
            tool=tool,
            config=config,
            mean=speaker_mean)  # for compute speaker sttdev
        # NOTE: input_data_dict_speaker have been not normalized yet

        for utt_index, input_utt in input_data_dict_speaker.items():

            if normalize == 'no':
                pass
            elif normalize == 'global' or not is_training:
                # Normalize by mean & std over the training set
                input_utt -= global_mean
                input_utt /= global_std
            elif normalize == 'speaker':
                # Normalize by mean & std per speaker
                input_utt = (input_utt - speaker_mean) / speaker_std
            elif normalize == 'utterance':
                # Normalize by mean & std per utterance
                utt_mean = np.mean(input_utt, axis=0, dtype=dtype)
                utt_std = np.std(input_utt, axis=0, dtype=dtype)
                input_utt = (input_utt - utt_mean) / utt_std
            else:
                ValueError

            frame_num_dict[speaker + '_' + utt_index] = input_utt.shape[0]

            if save_path is not None:
                # Save input features
                if save_format == 'numpy':
                    input_data_save_path = mkdir_join(
                        save_path, speaker, speaker + '_' + utt_index + '.npy')
                    np.save(input_data_save_path, input_utt)
                elif save_format == 'htk':
                    if sampPeriod is None:
                        _, sampPeriod, parmKind = read(audio_path)
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

    return global_mean, global_std, frame_num_dict
