#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Main function to make dataset (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import argparse
from glob import glob

sys.path.append('../')
from librispeech.prepare_path import Prepare
from librispeech.inputs.input_data import read_audio
# from librispeech.labels.phone import read_phone
from librispeech.labels.character import read_char
from librispeech.labels.word import read_word
from utils.util import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to Librispeech dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--run_root_path', type=str,
                    help='path to run this script')
parser.add_argument('--tool', type=str,
                    help='the tool to extract features, htk or python_speech_features or htk')
parser.add_argument('--htk_save_path', type=str, default='',
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
prep = Prepare(args.data_path, args.run_root_path)

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


def make_input(train_data_size):

    print('==================================================')
    print('  train_data_size: %s' % train_data_size)
    print('==================================================')

    input_save_path = mkdir_join(
        args.dataset_save_path, 'inputs', train_data_size)

    print('=> Processing input data...')
    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        print('---------- train ----------')
        if args.tool == 'htk':
            if train_data_size == 'train_all':
                audio_paths = [path for path in sorted(
                    glob(join(args.htk_save_path, 'train_clean100/*/*/*.htk')))]
                audio_paths += [path for path in sorted(
                    glob(join(args.htk_save_path, 'train_clean360/*/*/*.htk')))]
                audio_paths += [path for path in sorted(
                    glob(join(args.htk_save_path, 'train_other500/*/*/*.htk')))]
            else:
                audio_paths = [path for path in sorted(
                    glob(join(args.htk_save_path, train_data_size + '/*/*/*.htk')))]
            # NOTE: these are htk file paths
        else:
            audio_paths = prep.wav(data_type=train_data_size)

        # Read htk or wav files, and save input data and frame num dict
        train_global_mean_male, train_global_mean_female, train_global_std_male, train_global_std_female = read_audio(
            audio_paths=audio_paths,
            tool=args.tool,
            config=CONFIG,
            normalize=args.normalize,
            speaker_gender_dict=prep.speaker_gender_dict,
            is_training=True,
            save_path=mkdir_join(input_save_path, 'train'),)

        for data_type in ['dev_clean', 'dev_other', 'test_clean', 'test_other']:
            print('---------- %s ----------' % data_type)

            if args.tool == 'htk':
                audio_paths = [path for path in sorted(
                    glob(join(args.htk_save_path, data_type + '/*/*/*.htk')))]
            # NOTE: these are htk file paths
            else:
                audio_paths = prep.wav(data_type=data_type)

            # Read htk or wav files, and save input data and frame num dict
            read_audio(audio_paths=audio_paths,
                       tool=args.tool,
                       config=CONFIG,
                       normalize=args.normalize,
                       speaker_gender_dict=prep.speaker_gender_dict,
                       is_training=False,
                       save_path=mkdir_join(input_save_path, data_type),
                       train_global_mean_male=train_global_mean_male,
                       train_global_mean_female=train_global_mean_female,
                       train_global_std_male=train_global_std_female,
                       train_global_std_female=train_global_std_female)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'complete.txt'), 'w') as f:
            f.write('')


def make_label(model, train_data_size, label_type):

    print('==================================================')
    print('  model: %s' % model)
    print('  train_data_size: %s' % train_data_size)
    print('  label_type: %s' % label_type)
    print('==================================================')

    label_save_path = mkdir_join(
        args.dataset_save_path, 'labels', model, label_type, train_data_size)

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        save_map_file = True if train_data_size == 'train_all' else False
        divide_by_capital = True if label_type == 'character_capital_divide' else False

        # Read target labels and save labels as npy files
        print('---------- train ----------')
        label_paths = prep.text(data_type=train_data_size)
        read_char(label_paths=label_paths,
                  run_root_path=prep.run_root_path,
                  model=model,
                  save_map_file=save_map_file,
                  save_path=mkdir_join(label_save_path, 'train'),
                  divide_by_capital=divide_by_capital)

        for data_type in ['dev_clean', 'dev_other', 'test_clean', 'test_other']:

            # Read target labels and save labels as npy files
            print('---------- %s ----------' % data_type)
            read_char(label_paths=prep.text(data_type=data_type),
                      run_root_path=prep.run_root_path,
                      model=model,
                      save_path=mkdir_join(label_save_path, data_type),
                      divide_by_capital=divide_by_capital)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')


def make_label_word(model, train_data_size):

    print('==================================================')
    print('  model: %s' % model)
    print('  train_data_size: %s' % train_data_size)
    print('==================================================')

    label_save_path = mkdir_join(
        args.dataset_save_path, 'labels', model, 'word', train_data_size)

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        # Read target labels and save labels as npy files
        print('---------- train ----------')
        label_paths = prep.text(data_type=train_data_size)
        read_word(label_paths=label_paths,
                  data_type=train_data_size,
                  train_data_size=train_data_size,
                  run_root_path=prep.run_root_path,
                  model=model,
                  save_map_file=True,
                  save_path=mkdir_join(label_save_path, 'train'),
                  frequency_threshold=10)

        for data_type in ['dev_clean', 'dev_other', 'test_clean', 'test_other']:

            # Read target labels and save labels as npy files
            print('---------- %s ----------' % data_type)
            read_word(label_paths=prep.text(data_type=data_type),
                      data_type=data_type,
                      train_data_size=train_data_size,
                      run_root_path=prep.run_root_path,
                      model=model,
                      save_path=mkdir_join(label_save_path, data_type))

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':

    for train_data_size in ['train_all', 'train_clean100', 'train_clean360',
                            'train_other500']:
        # input
        make_input(train_data_size)

        # label
        for model in ['ctc', 'attention']:
            for label_type in ['character', 'character_capital_divide']:
                make_label(model, train_data_size, label_type)
                # TODO: add phone
            make_label_word(model, train_data_size)
