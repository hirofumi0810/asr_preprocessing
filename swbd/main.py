#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for the End-to-End model (Switchboard corpus).
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
from collections import Counter
import pickle

sys.path.append('../')
from swbd.path import Path
from swbd.input_data import read_audio
from swbd.labels.ldc97s62.character import read_trans
from swbd.labels.fisher.character import read_trans as read_trans_fisher
from swbd.labels.eval2000.stm import read_stm
from utils.util import mkdir_join
from utils.inputs.wav_split import split_wav
from utils.dataset import add_element

parser = argparse.ArgumentParser()
parser.add_argument('--swbd_audio_path', type=str,
                    help='path to LDC97S62 audio files')
parser.add_argument('--swbd_trans_path', type=str,
                    help='path to LDC97S62 transciption files')
parser.add_argument('--fisher_path', type=str, help='path to Fisher dataset')
parser.add_argument('--eval2000_audio_path', type=str,
                    help='path to audio files of eval2000 dataset')
parser.add_argument('--eval2000_trans_path', type=str,
                    help='path to transcript files of eval2000 dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
parser.add_argument('--run_root_path', type=str,
                    help='path to run this script')
parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa'])
parser.add_argument('--wav_save_path', type=str, help='path to wav files.')
parser.add_argument('--htk_save_path', type=str, help='path to htk files.')
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
parser.add_argument('--fisher', type=int,
                    help='If True, create large-size dataset (2000h).')

args = parser.parse_args()
path = Path(swbd_audio_path=args.swbd_audio_path,
            swbd_trans_path=args.swbd_trans_path,
            fisher_path=args.fisher_path,
            eval2000_audio_path=args.eval2000_audio_path,
            eval2000_trans_path=args.eval2000_trans_path,
            wav_save_path=args.wav_save_path,
            htk_save_path=args.htk_save_path,
            run_root_path='./')

CONFIG = {
    'feature_type': args.feature_type,
    'channels': args.channels,
    'sampling_rate': 8000,
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

    ########################################
    # labels
    ########################################
    print('=> Processing transcripts...')
    speaker_dict_dict = {}  # dict of speaker_dict
    print('---------- train ----------')
    if data_size == '300h':
        speaker_dict_dict['train'] = read_trans(
            label_paths=path.trans(corpus='swbd'),
            word_boundary_paths=path.word(corpus='swbd'),
            run_root_path='./',
            vocab_file_save_path=mkdir_join('./config/vocab_files'),
            save_vocab_file=True)
    elif data_size == '2000h':
        speaker_dict_a, char_set_a, char_capital_set_a, word_count_dict_a = read_trans_fisher(
            label_paths=path.trans(corpus='fisher'),
            target_speaker='A')
        speaker_dict_b, char_set_b, char_capital_set_b, word_count_dict_b = read_trans_fisher(
            label_paths=path.trans(corpus='fisher'),
            target_speaker='B')

        # Meage 2 dictionaries
        speaker_dict = merge_dicts([speaker_dict_a, speaker_dict_b])
        char_set = char_set_a | char_set_b
        char_capital_set = char_capital_set_a | char_capital_set_b
        word_count_dict_fisher = dict(
            Counter(word_count_dict_a) + Counter(word_count_dict_b))

        speaker_dict_dict['train'] = read_trans(
            label_paths=path.trans(corpus='swbd'),
            word_boundary_paths=path.word(corpus='swbd'),
            run_root_path='./',
            vocab_file_save_path=mkdir_join('./config/vocab_files'),
            save_vocab_file=True,
            speaker_dict_fisher=speaker_dict,
            char_set=char_set,
            char_capital_set=char_capital_set,
            word_count_dict=word_count_dict_fisher)
        del speaker_dict

    print('---------- eval2000 (swbd + ch) ----------')
    speaker_dict_dict['eval2000_swbd'], speaker_dict_dict['eval2000_ch'] = read_stm(
        stm_path=path.stm_path,
        pem_path=path.pem_path,
        glm_path=path.glm_path,
        run_root_path='./')

    ########################################
    # inputs
    ########################################
    print('\n=> Processing input data...')
    input_save_path = mkdir_join(
        args.feature_save_path, args.save_format, data_size)
    for data_type in ['train', 'eval2000_swbd', 'eval2000_ch']:
        print('---------- %s ----------' % data_type)
        if isfile(join(input_save_path, data_type, 'complete.txt')):
            print('Already exists.')
        else:
            if args.save_format == 'wav':
                ########################################
                # Split WAV files per utterance
                ########################################
                if data_type == 'train':
                    wav_paths = path.wav(corpus='swbd')
                    if data_size == '2000h':
                        wav_paths += path.wav(corpus='fisher')
                else:
                    wav_paths = path.wav(corpus=data_type)

                split_wav(wav_paths=wav_paths,
                          speaker_dict=speaker_dict_dict[data_type],
                          save_path=mkdir_join(input_save_path, data_type))
                # NOTE: ex.) save_path:
                # swbd/feature/save_format/data_size/data_type/speaker/utt_name.npy

            elif args.save_format in ['numpy', 'htk']:
                if data_type == 'train':
                    if args.tool == 'htk':
                        audio_paths = path.htk(corpus='swbd')
                        if data_size == '2000h':
                            audio_paths += path.htk(corpus='fisher')
                    else:
                        audio_paths = path.wav(corpus='swbd')
                        if data_size == '2000h':
                            audio_paths += path.wav(corpus='fisher')
                    is_training = True
                    global_mean, global_std = None, None
                else:
                    if args.tool == 'htk':
                        audio_paths = path.htk(corpus=data_type)
                    else:
                        audio_paths = path.wav(corpus=data_type)
                    is_training = False

                    # Load statistics over train dataset
                    global_mean = np.load(
                        join(input_save_path, 'train/global_mean.npy'))
                    global_std = np.load(
                        join(input_save_path, 'train/global_std.npy'))

                read_audio(audio_paths=audio_paths,
                           tool=args.tool,
                           config=CONFIG,
                           normalize=args.normalize,
                           speaker_dict=speaker_dict_dict[data_type],
                           is_training=is_training,
                           save_path=mkdir_join(input_save_path, data_type),
                           save_format=args.save_format,
                           global_mean=global_mean,
                           global_std=global_std)
                # NOTE: ex.) save_path:
                # swbd/feature/save_format/data_size/data_type/speaker/*.npy

            # Make a confirmation file to prove that dataset was saved
            # correctly
            with open(join(input_save_path, data_type, 'complete.txt'), 'w') as f:
                f.write('')

        ########################################
        # dataset (csv)
        ########################################
        print('\n=> Saving dataset files...')
        dataset_save_path = mkdir_join(
            args.dataset_save_path, args.save_format, data_size, data_type)

        print('---------- %s ----------' % data_type)
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
        speaker_dict = speaker_dict_dict[data_type]
        for speaker, utt_dict in tqdm(speaker_dict.items()):
            for utt_index, utt_info in utt_dict.items():
                if args.save_format == 'numpy':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, speaker + '_' + utt_index + '.npy')
                elif args.save_format == 'htk':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, speaker + '_' + utt_index + '.htk')
                elif args.save_format == 'wav':
                    input_utt_save_path = path.utt2wav(utt_index)
                else:
                    raise ValueError('save_format is numpy or htk or wav.')

                assert isfile(input_utt_save_path)
                frame_num = frame_num_dict[speaker + '_' + utt_index]

                char_indices, char_indices_capital, word_freq1_indices = utt_info[2:5]
                word_freq5_indices, word_freq10_indices, word_freq15_indices = utt_info[5:8]

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
                if utt_count == 10000:
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


def merge_dicts(dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


if __name__ == '__main__':

    data_sizes = ['2000h']
    # data_sizes = ['300h']
    # if bool(args.fisher):
    #     data_sizes += ['2000h']

    for data_size in data_sizes:
        main(data_size)
