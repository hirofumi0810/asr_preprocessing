#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Main function to make dataset (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile, abspath
import sys
import argparse

sys.path.append('../')
from timit.path import Path
from timit.transcript_character import read_char
from timit.transcript_phone import read_phone
from timit.input_data import read_audio
from utils.util import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--config_path', type=str,
                    help='path to config directory')

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
parser.add_argument('--energy', type=int, default=0,
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


def make_input():

    input_save_path = mkdir_join(args.dataset_save_path, 'inputs')

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
        train_global_mean_male, train_global_std_male, train_global_mean_female, train_global_std_female = read_audio(
            audio_paths=audio_paths,
            tool=args.tool,
            config=CONFIG,
            normalize=args.normalize,
            is_training=True,
            save_path=mkdir_join(input_save_path, 'train'))
        # NOTE: ex.) save_path: timit/inputs/train/***.npy

        for data_type in ['dev', 'test']:
            print('---------- %s ----------' % data_type)
            if args.tool == 'htk':
                audio_paths = path.htk(data_type=data_type)
            else:
                audio_paths = path.wav(data_type=data_type)

            # Read htk or wav files, and save input data and frame num dict
            read_audio(audio_paths=audio_paths,
                       tool=args.tool,
                       config=CONFIG,
                       normalize=args.normalize,
                       is_training=False,
                       save_path=mkdir_join(input_save_path, data_type),
                       train_global_mean_male=train_global_mean_male,
                       train_global_std_male=train_global_std_male,
                       train_global_mean_female=train_global_mean_female,
                       train_global_std_female=train_global_std_female)
            # NOTE: ex.) save_path: timit/inputs/data_type/***.npy

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'complete.txt'), 'w') as f:
            f.write('')


def make_label():

    label_save_path = mkdir_join(args.dataset_save_path, 'labels')

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.\n')
    else:
        for data_type in ['train', 'dev', 'test']:
            save_map_file = True if data_type == 'train' else False
            is_test = True if data_type == 'test' else False

            print('==================================================')
            print('  label_type: character, character_capital_divide')
            print('  data_type: %s' % data_type)
            print('==================================================')
            read_char(label_paths=path.trans(data_type=data_type),
                      map_file_save_path=mkdir_join(
                          abspath('./config'), 'mapping_files'),
                      is_test=is_test,
                      save_map_file=save_map_file,
                      save_path=mkdir_join(label_save_path, data_type))
            # NOTE: ex.) save_path:
            # timit/labels/data_type/character*/***.npy

            print('==================================================')
            print('  label_type: phone')
            print('  data_type: %s' % data_type)
            print('==================================================')
            read_phone(label_paths=path.phone(data_type=data_type),
                       map_file_save_path=mkdir_join(
                           abspath('./config'), 'mapping_files'),
                       is_test=is_test,
                       save_map_file=save_map_file,
                       save_path=mkdir_join(label_save_path, data_type))
            # NOTE: ex.) save_path:
            # timit/labels/data_type/phone**/***.npy

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')


def main():

    make_input()
    make_label()


if __name__ == '__main__':
    main()
