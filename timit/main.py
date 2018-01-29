#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Main function to make dataset (TIMIT corpus)."""

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
from timit.path import Path
from timit.transcript_character import read_char
from timit.transcript_phone import read_phone
from timit.input_data import read_audio
from utils.util import mkdir_join
from utils.dataset import add_element

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
parser.add_argument('--config_path', type=str, help='path to config directory')
parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa'])
parser.add_argument('--htk_save_path', type=str, help='path to save htk files')
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

args = parser.parse_args()
path = Path(data_path=args.data_path,
            config_path=args.config_path,
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


def main():

    for data_type in ['train', 'dev', 'test']:
        print('=' * 50)
        print(' ' * 20 + data_type + ' ' * 20)
        print('=' * 50)

        ########################################
        # inputs
        ########################################
        print('=> Processing input data...')
        if args.save_format in ['numpy', 'htk']:
            input_save_path = mkdir_join(
                args.feature_save_path, args.save_format)
            if isfile(join(input_save_path, data_type, 'complete.txt')):
                print('Already exists.')
            else:
                if args.tool == 'htk':
                    audio_paths = path.htk(data_type=data_type)
                else:
                    audio_paths = path.wav(data_type=data_type)

                if data_type != 'train':
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
                else:
                    is_training = True
                    global_mean_male, global_std_male, global_mean_female, global_std_female = None, None, None, None

                # Read htk or wav files, and save input data and frame num dict
                read_audio(audio_paths=audio_paths,
                           tool=args.tool,
                           config=CONFIG,
                           normalize=args.normalize,
                           is_training=is_training,
                           save_path=mkdir_join(input_save_path, data_type),
                           save_format=args.save_format,
                           global_mean_male=global_mean_male,
                           global_std_male=global_std_male,
                           global_mean_female=global_mean_female,
                           global_std_female=global_std_female)
                # NOTE: ex.) save_path:
                # timit/feature/save_format/data_type/*.npy

            # Make a confirmation file to prove that dataset was saved
            # correctly
            with open(join(input_save_path, data_type, 'complete.txt'), 'w') as f:
                f.write('')

        ########################################
        # labels (character)
        ########################################
        print('\n=> Processing transcripts (char)...')
        save_vocab_file = True if data_type == 'train' else False
        is_test = True if data_type == 'test' else False
        trans_dict = read_char(
            label_paths=path.trans(data_type=data_type),
            vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
            save_vocab_file=save_vocab_file,
            is_test=is_test)

        ########################################
        # dataset (character, csv)
        ########################################
        print('\n=> Saving dataset files (char)...')
        dataset_save_path = mkdir_join(
            args.dataset_save_path, args.save_format, data_type)
        df_columns = ['frame_num', 'input_path', 'transcript']
        df_char = pd.DataFrame([], columns=df_columns)
        df_char_capital = pd.DataFrame([], columns=df_columns)

        with open(join(input_save_path, data_type, 'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        for utt_name, trans_list in tqdm(trans_dict.items()):
            if args.save_format == 'numpy':
                speaker = utt_name.split('_')[0]
                input_utt_save_path = join(
                    input_save_path, data_type, speaker, utt_name + '.npy')
            elif args.save_format == 'htk':
                speaker = utt_name.split('_')[0]
                input_utt_save_path = join(
                    input_save_path, data_type, speaker, utt_name + '.htk')
            elif args.save_format == 'wav':
                input_utt_save_path = path.utt2wav(utt_name)
            else:
                raise ValueError('save_format is numpy or htk or wav.')

            assert isfile(input_utt_save_path)
            frame_num = frame_num_dict[utt_name]

            char_indices, char_indices_capital = trans_list

            df_char = add_element(
                df_char, [frame_num, input_utt_save_path, char_indices])
            df_char_capital = add_element(
                df_char_capital, [frame_num, input_utt_save_path, char_indices_capital])

        df_char.to_csv(join(dataset_save_path, 'character.csv'))
        df_char_capital.to_csv(
            join(dataset_save_path, 'character_capital_divide.csv'))

        ########################################
        # labels (phone)
        ########################################
        print('\n=> Processing transcripts (phone)...')
        trans_dict = read_phone(
            label_paths=path.phone(data_type=data_type),
            vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
            save_vocab_file=save_vocab_file,
            is_test=is_test)

        ########################################
        # dataset (phone, csv)
        ########################################
        print('\n=> Saving dataset files (phone)...')
        df_phone61 = pd.DataFrame([], columns=df_columns)
        df_phone48 = pd.DataFrame([], columns=df_columns)
        df_phone39 = pd.DataFrame([], columns=df_columns)

        for utt_name, trans_list in tqdm(trans_dict.items()):
            if args.save_format == 'numpy':
                speaker = utt_name.split('_')[0]
                input_utt_save_path = join(
                    input_save_path, data_type, speaker, utt_name + '.npy')
            elif args.save_format == 'htk':
                speaker = utt_name.split('_')[0]
                input_utt_save_path = join(
                    input_save_path, data_type, speaker, utt_name + '.htk')
            elif args.save_format == 'wav':
                input_utt_save_path = path.utt2wav(utt_name)
            else:
                raise ValueError('save_format is numpy or htk or wav.')

            assert isfile(input_utt_save_path)
            frame_num = frame_num_dict[utt_name]

            phone61_indices, phone48_indices, phone39_indices = trans_list

            df_phone61 = add_element(
                df_phone61, [frame_num, input_utt_save_path, phone61_indices])
            df_phone48 = add_element(
                df_phone48, [frame_num, input_utt_save_path, phone48_indices])
            df_phone39 = add_element(
                df_phone39, [frame_num, input_utt_save_path, phone39_indices])

        df_phone61.to_csv(join(dataset_save_path, 'phone61.csv'))
        df_phone48.to_csv(join(dataset_save_path, 'phone48.csv'))
        df_phone39.to_csv(join(dataset_save_path, 'phone39.csv'))


if __name__ == '__main__':
    main()
