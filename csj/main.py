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

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to CSJ dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
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
parser.add_argument('--sampling_rate', type=int,
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
parser.add_argument('--subset', type=str,
                    help='If True, create small dataset.')
parser.add_argument('--fullset', type=str,
                    help='If True, create full-size dataset.')

args = parser.parse_args()
path = Path(data_path=args.data_path,
            config_path='./config',
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


def main(data_size):

    print('=' * 50)
    print('  data_size: %s' % data_size)
    print('=' * 50)

    print('=> Processing transcripts...')
    if isfile(join(args.dataset_save_path, data_size, 'complete.txt')):
        print('Already exists.')
    else:
        ####################
        # labels
        ####################
        speaker_dict_dict = {}  # dict of speaker_dict

        for data_type in ['train', 'dev', 'eval1', 'eval2', 'eval3']:
            print('---------- %s ----------' % data_type)
            if data_type == 'train':
                label_paths = path.trans(data_type='train_' + data_size)
            else:
                label_paths = path.trans(data_type=data_type)
            save_vocab_file = True if data_type == 'train' else False
            is_test = True if 'eval' in data_type else False

            speaker_dict_dict[data_type] = read_sdb(
                label_paths=label_paths,
                data_size=data_size,
                vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
                save_vocab_file=save_vocab_file,
                is_test=is_test,
                data_type=data_type)

        ####################
        # inputs
        ####################
        input_save_path = mkdir_join(args.feature_save_path, data_size)
        print('=> Processing input data...')
        # NOTE: feature segmentation depends on transcriptions
        if isfile(join(input_save_path, 'complete.txt')):
            print('Already exists.')
        else:
            print('---------- train ----------')
            if args.tool == 'htk':
                audio_paths = path.htk(data_type='train_' + data_size)
            else:
                audio_paths = path.wav(data_type='train_' + data_size)

            global_mean_male, global_mean_female, global_std_male, global_std_female = read_audio(
                audio_paths=audio_paths,
                speaker_dict=speaker_dict_dict['train'],
                tool=args.tool,
                config=CONFIG,
                normalize=args.normalize,
                is_training=True,
                save_path=mkdir_join(input_save_path, 'train'))
            # NOTE: ex.) save_path:
            # csj/feature/data_size/train/speaker/*.npy

            for data_type in ['dev', 'eval1', 'eval2', 'eval3']:
                print('---------- %s ----------' % data_type)
                if args.tool == 'htk':
                    audio_paths = path.htk(data_type=data_type)
                else:
                    audio_paths = path.wav(data_type=data_type)

                read_audio(
                    audio_paths=audio_paths,
                    speaker_dict=speaker_dict_dict[data_type],
                    tool=args.tool,
                    config=CONFIG,
                    normalize=args.normalize,
                    is_training=False,
                    save_path=mkdir_join(input_save_path, data_type),
                    global_mean_male=global_mean_male,
                    global_std_male=global_std_male,
                    global_mean_female=global_mean_female,
                    global_std_female=global_std_female)
                # NOTE: ex.) save_path:
                # csj/feature/data_size/data_type/speaker/*.npy

            # Make a confirmation file to prove that dataset was saved
            # correctly
            with open(join(input_save_path, 'complete.txt'), 'w') as f:
                f.write('')

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

            for speaker, utt_dict in tqdm(speaker_dict_dict[data_type].items()):
                for utt_name, utt_info in utt_dict.items():
                    trans_kanji, trans_kana, trans_phone = utt_info[2:]

                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, speaker + '_' + utt_name + '.npy')
                    assert isfile(input_utt_save_path)
                    print(input_utt_save_path)
                    input_utt = np.load(input_utt_save_path)
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

                df_kanji.to_csv(join(dataset_save_path, 'dataset_kanji.csv'))
                df_kana.to_csv(join(dataset_save_path, 'dataset_kana.csv'))
                df_phone.to_csv(join(dataset_save_path, 'dataset_phone.csv'))

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(args.dataset_save_path, data_size, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':

    data_sizes = []
    if bool(args.subset):
        data_sizes += ['subset']
    if bool(args.fullset):
        data_sizes += ['fullset']

    for data_size in data_sizes:
        main(data_size)
