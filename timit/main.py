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

sys.path.append('../')
from timit.path import Path
from timit.transcript_character import read_char
from timit.transcript_phone import read_phone
from timit.input_data import read_audio
from utils.util import mkdir_join, mkdir

from utils.inputs.htk import read
from utils.inputs.wav2feature_python_speech_features import wav2feature as w2f_psf

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
parser.add_argument('--config_path', type=str, help='path to config directory')
parser.add_argument('--tool', type=str,
                    help='htk or python_speech_features or htk')
parser.add_argument('--htk_save_path', type=str, help='path to save htk files')
parser.add_argument('--normalize', type=str,
                    help='global (per gender) or speaker or utterance or no')
parser.add_argument('--save_format', type=str, help='numpy or htk or wav')

parser.add_argument('--feature_type', type=str, help='fbank or mfcc')
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

        ########################################
        # inputs
        ########################################
        print('=> Processing input data...')
        if args.save_format in ['numpy', 'htk']:
            input_save_path = mkdir_join(
                args.feature_save_path, args.save_format)
            if isfile(join(input_save_path, 'complete.txt')):
                print('Already exists.\n')
            else:
                print('---------- %s ----------' % data_type)
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
            with open(join(input_save_path, 'complete.txt'), 'w') as f:
                f.write('')

        ########################################
        # labels (character)
        ########################################
        print('=> Processing transcripts...')
        dataset_save_path = mkdir_join(args.dataset_save_path, data_type)
        save_vocab_file = True if data_type == 'train' else False
        print('=' * 50)
        print('  label_type: character, character_capital_divide')
        print('=' * 50)
        trans_dict = read_char(
            label_paths=path.trans(data_type=data_type),
            vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
            save_vocab_file=save_vocab_file)

        ########################################
        # dataset (character, csv)
        ########################################
        df_char = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])
        for utt_name, trans_char in tqdm(trans_dict.items()):
            if args.save_format == 'numpy':
                speaker = utt_name.split('_')[0]
                input_utt_save_path = join(
                    input_save_path, data_type, speaker, utt_name + '.npy')
                assert isfile(input_utt_save_path)
                input_utt = np.load(input_utt_save_path)
            elif args.save_format == 'htk':
                speaker = utt_name.split('_')[0]
                input_utt_save_path = join(
                    input_save_path, data_type, speaker, utt_name + '.htk')
                assert isfile(input_utt_save_path)
                input_utt, _, _ = read(input_utt_save_path)
            elif args.save_format == 'wav':
                input_utt_save_path = path.utt2wav(utt_name)
                assert isfile(input_utt_save_path)
                input_utt = w2f_psf(
                    input_utt_save_path,
                    feature_type=CONFIG['feature_type'],
                    feature_dim=CONFIG['channels'],
                    use_energy=CONFIG['energy'],
                    use_delta1=CONFIG['delta'],
                    use_delta2=CONFIG['deltadelta'],
                    window=CONFIG['window'],
                    slide=CONFIG['slide'])
            else:
                raise ValueError('save_format is numpy or htk or wav.')
            frame_num = input_utt.shape[0]

            series_char = pd.Series(
                [frame_num, input_utt_save_path, trans_char],
                index=df_char.columns)

            df_char = df_char.append(series_char, ignore_index=True)

        df_char.to_csv(
            join(dataset_save_path, 'dataset_' + args.save_format + '_character.csv'))

        ########################################
        # labels (phone)
        ########################################
        print('=' * 50)
        print('  label_type: phone')
        print('=' * 50)
        trans_dict = read_phone(
            label_paths=path.phone(data_type=data_type),
            vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
            save_vocab_file=save_vocab_file)

        ########################################
        # dataset (phone, csv)
        ########################################
        df_phone61 = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])
        df_phone48 = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])
        df_phone39 = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])
        for utt_name, [trans_phone61, trans_phone48, trans_phone39] in tqdm(trans_dict.items()):
            if args.save_format == 'numpy':
                speaker = utt_name.split('_')[0]
                input_utt_save_path = join(
                    input_save_path, data_type, speaker, utt_name + '.npy')
                assert isfile(input_utt_save_path)
                input_utt = np.load(input_utt_save_path)
            elif args.save_format == 'htk':
                speaker = utt_name.split('_')[0]
                input_utt_save_path = join(
                    input_save_path, data_type, speaker, utt_name + '.htk')
                assert isfile(input_utt_save_path)
                input_utt, _, _ = read(input_utt_save_path)
            elif args.save_format == 'wav':
                input_utt_save_path = path.utt2wav(utt_name)
                assert isfile(input_utt_save_path)
                input_utt = w2f_psf(
                    input_utt_save_path,
                    feature_type=CONFIG['feature_type'],
                    feature_dim=CONFIG['channels'],
                    use_energy=CONFIG['energy'],
                    use_delta1=CONFIG['delta'],
                    use_delta2=CONFIG['deltadelta'],
                    window=CONFIG['window'],
                    slide=CONFIG['slide'])
            else:
                raise ValueError('save_format is numpy or htk or wav.')
            frame_num = input_utt.shape[0]

            series_phone61 = pd.Series(
                [frame_num, input_utt_save_path, trans_phone61],
                index=df_phone61.columns)
            series_phone48 = pd.Series(
                [frame_num, input_utt_save_path, trans_phone48],
                index=df_phone48.columns)
            series_phone39 = pd.Series(
                [frame_num, input_utt_save_path, trans_phone39],
                index=df_phone39.columns)

            df_phone61 = df_phone61.append(
                series_phone61, ignore_index=True)
            df_phone48 = df_phone48.append(
                series_phone48, ignore_index=True)
            df_phone39 = df_phone39.append(
                series_phone39, ignore_index=True)

        df_phone61.to_csv(
            join(dataset_save_path, 'dataset_' + args.save_format + '_phone61.csv'))
        df_phone48.to_csv(
            join(dataset_save_path, 'dataset_' + args.save_format + '_phone48.csv'))
        df_phone39.to_csv(
            join(dataset_save_path, 'dataset_' + args.save_format + '_phone39.csv'))


if __name__ == '__main__':
    main()
