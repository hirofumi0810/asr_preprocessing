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

sys.path.append('../')
from swbd.path import Path
from swbd.input_data import read_audio
from swbd.labels.ldc97s62.character import read_trans
from swbd.labels.fisher.character import read_trans as read_trans_fisher
from swbd.labels.eval2000.stm import read_stm
from utils.util import mkdir_join

from utils.inputs.wav_split import split_wav
from utils.inputs.htk import read
from utils.inputs.wav2feature_python_speech_features import wav2feature as w2f_psf

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
                    help='htk or python_speech_features or htk')
parser.add_argument('--wav_save_path', type=str, help='path to wav files.')
parser.add_argument('--htk_save_path', type=str, help='path to htk files.')
parser.add_argument('--normalize', type=str,
                    help='global or speaker or utterance or None')
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
        speaker_dict_train = read_trans(
            label_paths=path.trans(corpus='swbd'),
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
        speaker_dict_fisher = merge_dicts([speaker_dict_a, speaker_dict_b])
        char_set = char_set_a | char_set_b
        char_capital_set = char_capital_set_a | char_capital_set_b
        word_count_dict_fisher = dict(
            Counter(word_count_dict_a) + Counter(word_count_dict_b))

        read_trans(
            label_paths=path.trans(corpus='swbd'),
            run_root_path='./',
            vocab_file_save_path=mkdir_join('./config/vocab_files'),
            save_vocab_file=True,
            speaker_dict_fisher=speaker_dict_fisher,
            char_set=char_set,
            char_capital_set=char_capital_set,
            word_count_dict=word_count_dict_fisher)
    speaker_dict_dict['train'] = speaker_dict_train

    print('---------- eval2000 (swbd + ch) ----------')
    speaker_dict_eval2000_swbd, speaker_dict_eval2000_ch = read_stm(
        stm_path=path.stm_path,
        pem_path=path.pem_path,
        glm_path=path.glm_path,
        run_root_path='./')
    speaker_dict_dict['eval2000_swbd'] = speaker_dict_eval2000_swbd
    speaker_dict_dict['eval2000_ch'] = speaker_dict_eval2000_ch

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
    for data_type in ['train', 'eval2000_swbd', 'eval2000_ch']:
        dataset_save_path = mkdir_join(
            args.dataset_save_path, args.save_format, data_size, data_type)

        print('---------- %s ----------' % data_type)
        df = pd.DataFrame(
            [], columns=['frame_num', 'input_path', 'transcript'])

        utt_count = 0
        df_list = []
        for speaker, utt_dict in tqdm(speaker_dict_dict[data_type].items()):
            for utt_index, utt_info in utt_dict.items():
                transcript = utt_info[2]
                if args.save_format == 'numpy':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, speaker + '_' + utt_index + '.npy')
                    assert isfile(input_utt_save_path)
                    input_utt = np.load(input_utt_save_path)
                elif args.save_format == 'htk':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, speaker + '_' + utt_index + '.htk')
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

        # Use the first 4000 utterances as the dev set
        if data_type == 'train':
            df[:4000].to_csv(mkdir_join(
                args.dataset_save_path, args.save_format, data_size, 'dev',
                'dataset.csv'))


def merge_dicts(dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


if __name__ == '__main__':

    data_sizes = ['300h']
    if bool(args.fisher):
        data_sizes += ['2000h']

    for data_size in data_sizes:
        main(data_size)
