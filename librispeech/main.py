#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Main function to make dataset (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

sys.path.append('../')
from librispeech.path import Path
from librispeech.input_data import read_audio
from librispeech.transcript import read_trans
from utils.util import mkdir_join
from utils.dataset import add_element

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to Librispeech dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa'])
parser.add_argument('--htk_save_path', type=str, help='path to save features')
parser.add_argument('--normalize', type=str,
                    choices=['global', 'speaker', 'utterance', 'no'])
parser.add_argument('--save_format', type=str, choices=['numpy', 'htk', 'wav'])

parser.add_argument('--feature_type', type=str, choices=['fbank', 'mfcc'])
parser.add_argument('--channels', type=int,
                    help='the number of frequency channels')
parser.add_argument('--window', type=float,
                    help='window width to extract features')
parser.add_argument('--slide', type=float, help='extract features per slide')
parser.add_argument('--energy', type=int, help='if 1, add the energy feature')
parser.add_argument('--delta', type=int, help='if 1, add the energy feature')
parser.add_argument('--deltadelta', type=int,
                    help='if 1, double delta features are also extracted')
parser.add_argument('--medium', type=int,
                    help='If True, create medium-size dataset (460h).')
parser.add_argument('--large', type=int,
                    help='If True, create large-size dataset (960h).')

args = parser.parse_args()
path = Path(data_path=args.data_path,
            htk_save_path=args.htk_save_path)

CONFIG = {
    'feature_type': args.feature_type,
    'channels': args.channels,
    'sampling_rate': 16000,
    'window': args.window,
    'slide': args.slide,
    'energy': bool(args.energy),
    'delta': bool(args.delta),
    'deltadelta': bool(args.deltadelta)
}

if args.save_format == 'htk':
    assert args.tool == 'htk'


def main(data_size):

    for data_type in ['train', 'dev_clean', 'dev_other', 'test_clean', 'test_other']:
        print('=' * 50)
        print(' ' * 20 + data_type + ' (' + data_size + ')' + ' ' * 20)
        print('=' * 50)

        ########################################
        # inputs
        ########################################
        print('=> Processing input data...')
        if args.save_format in ['numpy', 'htk']:
            input_save_path = mkdir_join(
                args.feature_save_path, args.save_format, data_size)
            if isfile(join(input_save_path, data_type, 'complete.txt')):
                print('Already exists.')
            else:
                if data_type == 'train':
                    if args.tool == 'htk':
                        audio_paths = path.htk(data_type='train' + data_size)
                    else:
                        audio_paths = path.wav(data_type='train' + data_size)
                    is_training = True
                    global_mean_male, global_std_male, global_mean_female, global_std_female = None, None, None, None
                else:
                    if args.tool == 'htk':
                        audio_paths = path.htk(data_type=data_type)
                    else:
                        audio_paths = path.wav(data_type=data_type)
                    is_training = False

                    # Load statistics over train dataset
                    global_mean_male = np.load(
                        join(input_save_path, 'train/global_mean_male.npy'))
                    global_std_male = np.load(
                        join(input_save_path, 'train/global_std_male.npy'))
                    global_mean_female = np.load(
                        join(input_save_path, 'train/global_mean_female.npy'))
                    global_std_female = np.load(
                        join(input_save_path, 'train/global_std_female.npy'))

                read_audio(audio_paths=audio_paths,
                           tool=args.tool,
                           config=CONFIG,
                           normalize=args.normalize,
                           speaker_gender_dict=path.speaker_gender_dict,
                           is_training=is_training,
                           save_path=mkdir_join(input_save_path, data_type),
                           save_format=args.save_format,
                           global_mean_male=global_mean_male,
                           global_mean_female=global_mean_female,
                           global_std_male=global_std_male,
                           global_std_female=global_std_female)
                # NOTE: ex.) save_path:
                # librispeech/feature/save_format/data_size/data_type/speaker/*.npy

            # Make a confirmation file to prove that dataset was saved
            # correctly
            with open(join(input_save_path, data_type, 'complete.txt'), 'w') as f:
                f.write('')

        ########################################
        # labels
        ########################################
        print('\n=> Processing transcripts...')
        if data_type == 'train':
            label_paths = path.trans(data_type='train' + data_size)
        else:
            label_paths = path.trans(data_type=data_type)
        save_vocab_file = True if data_type == 'train' else False
        is_test = True if 'test' in data_type else False

        speaker_dict = read_trans(
            label_paths=label_paths,
            data_size=data_size,
            vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
            save_vocab_file=save_vocab_file,
            is_test=is_test,
            data_type=data_type)

        ########################################
        # dataset (csv)
        ########################################
        print('\n=> Saving dataset files...')
        dataset_save_path = mkdir_join(
            args.dataset_save_path, args.save_format, data_size, data_type)
        df_columns = ['frame_num', 'input_path', 'transcript']
        df_char = pd.DataFrame([], columns=df_columns)
        df_char_capital = pd.DataFrame([], columns=df_columns)
        df_word_freq1 = pd.DataFrame([], columns=df_columns)
        df_word_freq5 = pd.DataFrame([], columns=df_columns)
        df_word_freq10 = pd.DataFrame([], columns=df_columns)
        df_word_freq15 = pd.DataFrame([], columns=df_columns)

        with open(join(input_save_path, data_type, 'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        utt_count = 0
        df_char_list, df_char_capital_list = [], []
        df_word_freq1_list, df_word_freq5_list = [], []
        df_word_freq10_list, df_word_freq15_list = [], []
        for speaker, utt_dict in tqdm(speaker_dict.items()):
            for utt_name, indices_list in utt_dict.items():
                if args.save_format == 'numpy':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, utt_name + '.npy')
                elif args.save_format == 'htk':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, utt_name + '.htk')
                elif args.save_format == 'wav':
                    input_utt_save_path = path.utt2wav(utt_name)
                else:
                    raise ValueError('save_format is numpy or htk or wav.')

                assert isfile(input_utt_save_path)
                frame_num = frame_num_dict[utt_name]

                char_indices, char_indices_capital, word_freq1_indices = indices_list[:3]
                word_freq5_indices, word_freq10_indices, word_freq15_indices = indices_list[
                    3:6]

                df_char = add_element(
                    df_char, [frame_num, input_utt_save_path, char_indices])
                df_char_capital = add_element(
                    df_char_capital, [frame_num, input_utt_save_path, char_indices_capital])
                df_word_freq1 = add_element(
                    df_word_freq1, [frame_num, input_utt_save_path, word_freq1_indices])
                df_word_freq5 = add_element(
                    df_word_freq5, [frame_num, input_utt_save_path, word_freq5_indices])
                df_word_freq10 = add_element(
                    df_word_freq10, [frame_num, input_utt_save_path, word_freq10_indices])
                df_word_freq15 = add_element(
                    df_word_freq15, [frame_num, input_utt_save_path, word_freq15_indices])
                utt_count += 1

                # Reset
                if utt_count == 50000:
                    df_char_list.append(df_char)
                    df_char_capital_list.append(df_char_capital)
                    df_word_freq1_list.append(df_word_freq1)
                    df_word_freq5_list.append(df_word_freq5)
                    df_word_freq10_list.append(df_word_freq10)
                    df_word_freq15_list.append(df_word_freq15)

                    df_char = pd.DataFrame([], columns=df_columns)
                    df_char_capital = pd.DataFrame([], columns=df_columns)
                    df_word_freq1 = pd.DataFrame([], columns=df_columns)
                    df_word_freq5 = pd.DataFrame([], columns=df_columns)
                    df_word_freq10 = pd.DataFrame([], columns=df_columns)
                    df_word_freq15 = pd.DataFrame([], columns=df_columns)
                    utt_count = 0

        # Last dataframe
        df_char_list.append(df_char)
        df_char_capital_list.append(df_char_capital)
        df_word_freq1_list.append(df_word_freq1)
        df_word_freq5_list.append(df_word_freq5)
        df_word_freq10_list.append(df_word_freq10)
        df_word_freq15_list.append(df_word_freq15)

        # Concatenate all dataframes
        df_char = df_char_list[0]
        df_char_capital = df_char_capital_list[0]
        df_word_freq1 = df_word_freq1_list[0]
        df_word_freq5 = df_word_freq5_list[0]
        df_word_freq10 = df_word_freq10_list[0]
        df_word_freq15 = df_word_freq15_list[0]

        for df_i in df_char_list[1:]:
            df_char = pd.concat([df_char, df_i], axis=0)
        for df_i in df_char_list[1:]:
            df_char_capital = pd.concat([df_char_capital, df_i], axis=0)
        for df_i in df_word_freq1_list[1:]:
            df_word_freq1 = pd.concat([df_word_freq1, df_i], axis=0)
        for df_i in df_word_freq5_list[1:]:
            df_word_freq5 = pd.concat([df_word_freq5, df_i], axis=0)
        for df_i in df_word_freq10_list[1:]:
            df_word_freq10 = pd.concat([df_word_freq10, df_i], axis=0)
        for df_i in df_word_freq15_list[1:]:
            df_word_freq15 = pd.concat([df_word_freq15, df_i], axis=0)

        df_char.to_csv(join(dataset_save_path, 'character.csv'))
        df_char_capital.to_csv(
            join(dataset_save_path, 'character_capital_divide.csv'))
        df_word_freq1.to_csv(join(dataset_save_path, 'word_freq1.csv'))
        df_word_freq5.to_csv(join(dataset_save_path, 'word_freq5.csv'))
        df_word_freq10.to_csv(join(dataset_save_path, 'word_freq10.csv'))
        df_word_freq15.to_csv(join(dataset_save_path, 'word_freq15.csv'))


if __name__ == '__main__':

    data_sizes = ['100h']
    if bool(args.medium):
        data_sizes += ['460h']
    if bool(args.large):
        data_sizes += ['960h']

    for data_size in data_sizes:
        main(data_size)
        # TODO: add phone
