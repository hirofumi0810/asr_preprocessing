#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for the End-to-End model (CSJ corpus).
   Note that feature extraction depends on transcripts.
"""

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
from csj.path import Path
from csj.input_data import read_audio
from csj.labels.transcript import read_sdb
from utils.util import mkdir_join

from utils.inputs.wav_split import split_wav
from utils.inputs.htk import read
from utils.inputs.wav2feature_python_speech_features import wav2feature as w2f_psf

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to CSJ dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
parser.add_argument('--wav_save_path', type=str,
                    help='path to save wav files (per utterance)')
parser.add_argument('--tool', type=str,
                    help='htk or python_speech_features or htk')
parser.add_argument('--htk_save_path', type=str, help='path to save features')
parser.add_argument('--normalize', type=str,
                    help='global or speaker or utterance')
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
parser.add_argument('--subset', type=int,
                    help='If True, create small dataset.')
parser.add_argument('--fullset', type=int,
                    help='If True, create full-size dataset.')

args = parser.parse_args()
path = Path(data_path=args.data_path,
            config_path='./config',
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

    print('=' * 50)
    print('  data_size: %s' % data_size)
    print('=' * 50)

    speaker_dict_dict = {}  # dict of speaker_dict
    for data_type in ['train', 'eval1', 'eval2', 'eval3']:
        print('---------- %s ----------' % data_type)

        ########################################
        # labels
        ########################################
        if data_type == 'train':
            label_paths = path.trans(data_type='train_' + data_size)
        else:
            label_paths = path.trans(data_type=data_type)
        save_vocab_file = True if data_type == 'train' else False
        is_test = True if 'eval' in data_type else False

        print('=> Processing transcripts...')
        speaker_dict_dict[data_type] = read_sdb(
            label_paths=label_paths,
            data_size=data_size,
            vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
            save_vocab_file=save_vocab_file,
            is_test=is_test,
            data_type=data_type)

        ########################################
        # inputs
        ########################################
        print('=> Processing input data...')
        input_save_path = mkdir_join(
            args.feature_save_path, args.save_format, data_size)
        if isfile(join(input_save_path, data_type, 'complete.txt')):
            print('Already exists.')
        else:
            if args.save_format == 'wav':
                ########################################
                # Split WAV files per utterance
                ########################################
                if data_type == 'train':
                    wav_paths = path.wav(corpus='train' + data_size)
                else:
                    wav_paths = path.wav(corpus=data_type)

                split_wav(wav_paths=wav_paths,
                          speaker_dict=speaker_dict_dict[data_type],
                          save_path=mkdir_join(input_save_path, data_type))
                # NOTE: ex.) save_path:
                # csj/feature/save_format/data_size/data_type/speaker/utt_name.npy

            elif args.save_format in ['numpy', 'htk']:
                print('---------- %s ----------' % data_type)
                if data_type == 'train':
                    if args.tool == 'htk':
                        audio_paths = path.htk(data_type='train_' + data_size)
                    else:
                        audio_paths = path.wav(data_type='train_' + data_size)
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
                           speaker_dict=speaker_dict_dict[data_type],
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
                # csj/feature/save_format/data_size/data_type/speaker/*.npy

            # Make a confirmation file to prove that dataset was saved
            # correctly
            with open(join(input_save_path, data_type, 'complete.txt'), 'w') as f:
                f.write('')

    ########################################
    # dataset (csv)
    ########################################
    for data_type in ['train', 'dev', 'eval1', 'eval2', 'eval3']:
        dataset_save_path = mkdir_join(
            args.dataset_save_path, data_size, data_type)

        print('---------- %s ----------' % data_type)
        df_kanji = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])
        df_kana = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])
        df_phone = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])

        if data_type == 'dev':
            # Use the first 4000 utterances as the dev set
            utt_count = 0
            for speaker, utt_dict in tqdm(speaker_dict_dict['train'].items()):
                for utt_index, utt_info in utt_dict.items():
                    trans_kanji, trans_kana, trans_phone = utt_info[2:]
                    if args.save_format == 'numpy':
                        input_utt_save_path = join(
                            input_save_path, data_type, speaker, speaker + '_' + utt_index + '.npy')
                        assert isfile(input_utt_save_path)
                        input_utt = np.load(input_utt_save_path)
                    elif args.save_format == 'htk':
                        input_utt_save_path = join(
                            input_save_path, data_type, speaker, utt_index + '.htk')
                        assert isfile(input_utt_save_path)
                        input_utt, _, _ = read(input_utt_save_path)
                    elif args.save_format == 'wav':
                        input_utt_save_path = path.utt2wav(utt_index)
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

                    series_kanji = pd.Series(
                        [frame_num, input_utt_save_path, trans_kanji],
                        index=df_kanji.columns)
                    series_kana = pd.Series(
                        [frame_num, input_utt_save_path, trans_kana],
                        index=df_kana.columns)
                    series_phone = pd.Series(
                        [frame_num, input_utt_save_path, trans_phone],
                        index=df_phone.columns)

                    df_kanji = df_kanji.append(series_kanji, ignore_index=True)
                    df_kana = df_kana.append(series_kana, ignore_index=True)
                    df_phone = df_phone.append(series_phone, ignore_index=True)

                    utt_count += 1
                    if utt_count == 4000:
                        break
        else:
            for speaker, utt_dict in tqdm(speaker_dict_dict[data_type].items()):
                for utt_index, utt_info in utt_dict.items():
                    trans_kanji, trans_kana, trans_phone = utt_info[2:]
                    if args.save_format == 'numpy':
                        input_utt_save_path = join(
                            input_save_path, data_type, speaker, speaker + '_' + utt_index + '.npy')
                        assert isfile(input_utt_save_path)
                        input_utt = np.load(input_utt_save_path)
                    elif args.save_format == 'htk':
                        input_utt_save_path = join(
                            input_save_path, data_type, speaker, utt_index + '.htk')
                        assert isfile(input_utt_save_path)
                        input_utt, _, _ = read(input_utt_save_path)
                    elif args.save_format == 'wav':
                        input_utt_save_path = path.utt2wav(utt_index)
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

                    series_kanji = pd.Series(
                        [frame_num, input_utt_save_path, trans_kanji],
                        index=df_kanji.columns)
                    series_kana = pd.Series(
                        [frame_num, input_utt_save_path, trans_kana],
                        index=df_kana.columns)
                    series_phone = pd.Series(
                        [frame_num, input_utt_save_path, trans_phone],
                        index=df_phone.columns)

                    df_kanji = df_kanji.append(series_kanji, ignore_index=True)
                    df_kana = df_kana.append(series_kana, ignore_index=True)
                    df_phone = df_phone.append(series_phone, ignore_index=True)

            df_kanji.to_csv(
                join(dataset_save_path, 'dataset_' + args.save_format + '_kanji.csv'))
            df_kana.to_csv(
                join(dataset_save_path, 'dataset_' + args.save_format + '_kana.csv'))
            df_phone.to_csv(
                join(dataset_save_path, 'dataset_' + args.save_format + '_phone.csv'))


if __name__ == '__main__':

    data_sizes = []
    if bool(args.subset):
        data_sizes += ['subset']
    if bool(args.fullset):
        data_sizes += ['fullset']

    for data_size in data_sizes:
        main(data_size)
