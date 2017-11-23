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

sys.path.append('../')
from librispeech.path import Path
from librispeech.input_data import read_audio
from librispeech.transcript import read_trans
from utils.util import mkdir_join

from utils.inputs.htk import read
from utils.inputs.wav2feature_python_speech_features import wav2feature as w2f_psf

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to Librispeech dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
parser.add_argument('--tool', type=str,
                    help='htk or python_speech_features or htk')
parser.add_argument('--htk_save_path', type=str, help='path to save features')
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
        df = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])

        utt_count = 0
        df_list = []
        for speaker, utt_dict in tqdm(speaker_dict.items()):
            for utt_name, transcript in utt_dict.items():
                if args.save_format == 'numpy':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, utt_name + '.npy')
                    assert isfile(input_utt_save_path)
                    input_utt = np.load(input_utt_save_path)
                elif args.save_format == 'htk':
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
                del input_utt

                series = pd.Series(
                    [frame_num, input_utt_save_path, transcript],
                    index=df.columns)
                df = df.append(series, ignore_index=True)
                utt_count += 1

                # Reset
                if utt_count == 50000:
                    df_list.append(df)
                    df = pd.DataFrame(
                        [], columns=['frame_num', 'input_path', 'transcript'])
                    utt_count = 0

        # Last dataframe
        df_list.append(df)

        # Concatenate all dataframes
        df = df_list[0]
        for df_i in df_list[1:]:
            df = pd.concat([df, df_i], axis=0)

        df.to_csv(join(dataset_save_path, 'dataset.csv'))


if __name__ == '__main__':

    data_sizes = ['100h']
    if bool(args.medium):
        data_sizes += ['460h']
    if bool(args.large):
        data_sizes += ['960h']

    for data_size in data_sizes:
        main(data_size)
        # TODO: add phone
