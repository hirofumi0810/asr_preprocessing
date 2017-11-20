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

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
parser.add_argument('--config_path', type=str, help='path to config directory')
parser.add_argument('--tool', type=str,
                    help='the tool to extract features, htk or python_speech_features or htk')
parser.add_argument('--htk_save_path', type=str, default=None,
                    help='path to save features, this is needed only when you use HTK.')
parser.add_argument('--normalize', type=str, default='speaker',
                    help='global or speaker or utterance')

parser.add_argument('--feature_type', type=str, default='logmelfbank',
                    help='the type of features, logmelfbank or mfcc or linearmelfbank')
parser.add_argument('--channels', type=int, default=40,
                    help='the number of frequency channels')
parser.add_argument('--sampling_rate', type=float,
                    default=16000, help='sampling rate')
parser.add_argument('--window', type=float, default=0.025,
                    help='window width to extract features')
parser.add_argument('--slide', type=float, default=0.01,
                    help='extract features per \'slide\'')
parser.add_argument('--energy', type=int, default=1,
                    help='if 1, add the energy feature')
parser.add_argument('--delta', type=int, default=1,
                    help='if 1, add the energy feature')
parser.add_argument('--deltadelta', type=int, default=1,
                    help='if 1, double delta features are also extracted')

args = parser.parse_args()
path = Path(data_path=args.data_path,
            config_path=args.config_path,
            htk_save_path=args.htk_save_path)

CONFIG = {
    'feature_type': args.feature_type,
    'channels': args.channels,
    'sampling_rate': args.sampling_rate,
    'window': args.window,
    'slide': args.slide,
    'energy': bool(args.energy),
    'delta': bool(args.delta),
    'deltadelta': bool(args.deltadelta)
}


def main():

    input_save_path = mkdir(args.feature_save_path)

    print('=> Processing input data...')
    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.\n')
    else:
        print('---------- train ----------')
        if args.tool == 'htk':
            audio_paths = path.htk(data_type='train')
        else:
            audio_paths = path.wav(data_type='train')

        # Read htk or wav files, and save input data and frame num dict
        global_mean_male, global_std_male, global_mean_female, global_std_female = read_audio(
            audio_paths=audio_paths,
            tool=args.tool,
            config=CONFIG,
            normalize=args.normalize,
            is_training=True,
            save_path=mkdir_join(input_save_path, 'train'))
        # NOTE: ex.) save_path: timit/feature/train/*.npy

        for data_type in ['dev', 'test']:
            print('---------- %s ----------' % data_type)
            if args.tool == 'htk':
                audio_paths = path.htk(data_type=data_type)
            else:
                audio_paths = path.wav(data_type=data_type)

            # Read htk or wav files, and save input data and frame num dict
            read_audio(
                audio_paths=audio_paths,
                tool=args.tool,
                config=CONFIG,
                normalize=args.normalize,
                is_training=False,
                save_path=mkdir_join(input_save_path, data_type),
                global_mean_male=global_mean_male,
                global_std_male=global_std_male,
                global_mean_female=global_mean_female,
                global_std_female=global_std_female)
            # NOTE: ex.) save_path: timit/feature/data_type/*.npy

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'complete.txt'), 'w') as f:
            f.write('')

    print('=> Processing transcripts...')
    if isfile(join(args.dataset_save_path, 'complete.txt')):
        print('Already exists.\n')
    else:
        for data_type in ['train', 'dev', 'test']:
            dataset_save_path = mkdir_join(args.dataset_save_path, data_type)
            save_vocab_file = True if data_type == 'train' else False

            print('=' * 50)
            print('  label_type: character, character_capital_divide')
            print('  data_type: %s' % data_type)
            print('=' * 50)

            trans_dict = read_char(
                label_paths=path.trans(data_type=data_type),
                vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
                save_vocab_file=save_vocab_file)

            df_char = pd.DataFrame(
                [], columns=['frame_num', 'input_path', 'transcript'])
            for utt_name, trans_char in tqdm(trans_dict.items()):
                input_utt_save_path = join(
                    input_save_path, data_type, utt_name + '.npy')
                assert isfile(input_utt_save_path)
                input_utt = np.load(input_utt_save_path)
                frame_num = input_utt.shape[0]

                series_char = pd.Series(
                    [frame_num, input_utt_save_path, trans_char],
                    index=df_char.columns)

                df_char = df_char.append(series_char, ignore_index=True)

            df_char.to_csv(join(dataset_save_path, 'dataset_character.csv'))

            print('=' * 50)
            print('  label_type: phone')
            print('  data_type: %s' % data_type)
            print('=' * 50)

            trans_dict = read_phone(
                label_paths=path.phone(data_type=data_type),
                vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
                save_vocab_file=save_vocab_file)

            df_phone61 = pd.DataFrame(
                [], columns=['frame_num', 'input_path', 'transcript'])
            df_phone48 = pd.DataFrame(
                [], columns=['frame_num', 'input_path', 'transcript'])
            df_phone39 = pd.DataFrame(
                [], columns=['frame_num', 'input_path', 'transcript'])
            for utt_name, [trans_phone61, trans_phone48, trans_phone39] in tqdm(trans_dict.items()):
                input_utt_save_path = join(
                    input_save_path, data_type, utt_name + '.npy')
                assert isfile(input_utt_save_path)
                input_utt = np.load(input_utt_save_path)
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

            df_phone61.to_csv(join(dataset_save_path, 'dataset_phone61.csv'))
            df_phone48.to_csv(join(dataset_save_path, 'dataset_phone48.csv'))
            df_phone39.to_csv(join(dataset_save_path, 'dataset_phone39.csv'))

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(args.dataset_save_path, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':
    main()
