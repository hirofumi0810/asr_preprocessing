#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (Switchboard corpus)."""

from os.path import join, basename
import pickle
import numpy as np
from tqdm import tqdm

from utils.util import mkdir_join
from utils.inputs.segmentation import segment
# TODO: add segmentation ver.


def read_audio(audio_paths, speaker_dict, tool, config, normalize, is_training,
               save_path=None, global_mean=None, global_std=None,
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
            global => normalize input features by global mean & std over
                      the training set per gender
            speaker => normalize input features by mean & std per speaker
            utterance => normalize input features by mean & std per utterancet
                         data by mean & std per utterance
        is_training (bool): training or not
        save_path (string): path to save npy files
        global_mean (np.ndarray, optional): global mean over the training set
        global_std (np.ndarray, optional): global standard deviation over the
            training set
        dtype (optional): the type of data, default is np.float32
    Returns:
        global_mean (np.ndarray): global mean over the training set
        global_std (np.ndarray): global standard deviation over the
            training set
    """
    if not is_training:
        if global_mean is None or global_std is None:
            raise ValueError('Set mean & std computed in the training set.')
    if normalize not in ['global', 'speaker', 'utterance']:
        raise ValueError('normalize is "utterance" or "speaker" or "global".')

    total_frame_num = 0
    total_frame_num_dict = {}
    speaker_mean_dict = {}

    # Loop 1: Computing global mean and statistics
    if is_training:
        print('===> Reading audio files...')
        for i, audio_path in enumerate(tqdm(audio_paths)):
            speaker = basename(audio_path).split('.')[0]

            # Fix speaker name
            speaker = speaker.replace('sw0', 'sw')
            # ex.) sw04771A => sw4771-A (LDC97S62)
            speaker = speaker.replace('sw_', 'sw')
            # ex.) sw_4771A => sw4771-A (eval2000)

            # Divide each audio file into utterances
            _, input_data_utt_sum, speaker_mean, _, total_frame_num_speaker = segment(
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
                global_mean = np.zeros((feature_dim,), dtype=dtype)
                global_std = np.zeros((feature_dim,), dtype=dtype)

            global_mean += input_data_utt_sum
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
            # ex.) sw04771A => sw4771A (LDC97S62)
            speaker = speaker.replace('sw_', 'sw')
            # ex.) sw_4771A => sw4771A (eval2000)

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
            for input_data_utt in input_data_dict_speaker.values():
                global_std += np.sum(
                    np.abs(input_data_utt - global_mean) ** 2, axis=0)

        # Compute global stddev
        global_std = np.sqrt(global_std / (total_frame_num - 1))

        if save_path is not None:
            # Save global mean & std per gender
            np.save(join(save_path, 'global_mean.npy'), global_mean)
            np.save(join(save_path, 'global_std.npy'), global_std)

    # Loop 2: Normalization and Saving
    print('===> Normalization...')
    frame_num_dict = {}
    for audio_path in tqdm(audio_paths):
        speaker = basename(audio_path).split('.')[0]

        # Normalize speaker name
        speaker = speaker.replace('sw0', 'sw')
        # ex.) sw04771A => sw4771A (LDC97S62)
        speaker = speaker.replace('sw_', 'sw')
        # ex.) sw_4771A => sw4771A (eval2000)

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
                # Normalize by mean & std over the training set
                input_data_utt -= global_mean
                input_data_utt /= global_std

            if save_path is not None:
                # Save input features
                input_data_save_path = mkdir_join(
                    save_path, speaker, speaker + '_' + utt_index + '.npy')
                np.save(input_data_save_path, input_data_utt)
                frame_num_dict[speaker + '_' +
                               utt_index] = input_data_utt.shape[0]

    if save_path is not None:
        # Save the frame number dictionary
        with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
            pickle.dump(frame_num_dict, f)

    return global_mean, global_std
