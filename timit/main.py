#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Main function to make dataset (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile, abspath
import sys
import argparse
from glob import glob

sys.path.append('../')
from timit.prepare_path import Prepare
from timit.labels.character import read_text
from timit.labels.phone import read_phone
from timit.inputs.input_data import read_audio
from utils.util import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
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


def make_input():

    input_save_path = mkdir_join(
        args.dataset_save_path, 'inputs', args.tool, args.normalize)

    print('=> Processing input data...')
    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.\n')
    else:
        print('---------- train ----------')
        if args.tool == 'htk':
            audio_paths = [path for path in sorted(
                glob(join(args.htk_save_path, 'train/*/*.htk')))]
            # NOTE: these are htk file paths
        else:
            audio_paths = prep.wav(data_type='train')

        # Read htk or wav files, and save input data and frame num dict
        train_global_mean_male, train_global_std_male, train_global_mean_female, train_global_std_female = read_audio(
            audio_paths=audio_paths,
            tool=args.tool,
            config=CONFIG,
            normalize=args.normalize,
            is_training=True,
            save_path=mkdir_join(input_save_path, 'train'))

        for data_type in ['dev', 'test']:
            print('---------- %s ----------' % data_type)
            if args.tool == 'htk':
                audio_paths = [path for path in sorted(
                    glob(join(args.htk_save_path, data_type, '*/*.htk')))]
                # NOTE: these are htk file paths
            else:
                audio_paths = prep.wav(data_type=data_type)

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

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'complete.txt'), 'w') as f:
            f.write('')


def make_label(model, label_type):

    print('==================================================')
    print('  model: %s' % model)
    print('  label_type: %s' % label_type)
    print('==================================================')

    label_save_path = mkdir_join(
        args.dataset_save_path, 'labels', model, label_type)

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.\n')
    else:
        for data_type in ['train', 'dev', 'test']:
            save_map_file = True if data_type == 'train' else False

            # Read target labels and save labels as npy files
            print('---------- %s ----------' % data_type)
            if label_type in ['character', 'character_capital_divide']:
                divide_by_capital = False if label_type == 'character' else True
                read_text(label_paths=prep.text(data_type=data_type),
                          run_root_path=abspath('./'),
                          model=model,
                          save_map_file=save_map_file,
                          save_path=mkdir_join(label_save_path, data_type),
                          divide_by_capital=divide_by_capital)
            else:
                # 39 or 48 or 61 phones
                read_phone(label_paths=prep.phone(data_type=data_type),
                           label_type=label_type,
                           run_root_path=abspath('./'),
                           model=model,
                           save_map_file=save_map_file,
                           save_path=mkdir_join(label_save_path, data_type))

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')


def main():

    make_input()

    for model in ['ctc', 'attention']:
        for label_type in ['character', 'character_capital_divide',
                           'phone61', 'phone48', 'phone39']:
            make_label(model, label_type)


if __name__ == '__main__':
    main()
