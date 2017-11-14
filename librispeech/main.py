#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Main function to make dataset (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import argparse

sys.path.append('../')
from librispeech.path import Path
from librispeech.input_data import read_audio
from librispeech.transcript import read_trans
from utils.util import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to Librispeech dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
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
parser.add_argument('--medium', type=str,
                    help='If True, create medium-size dataset.')
parser.add_argument('--large', type=str,
                    help='If True, create large-size dataset.')

args = parser.parse_args()
path = Path(data_path=args.data_path,
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


def make_input(data_size):

    print('=' * 30)
    print('  data_size: %s' % data_size)
    print('=' * 30)

    input_save_path = mkdir_join(
        args.dataset_save_path, 'inputs', data_size)

    print('=> Processing input data...')
    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        print('---------- train ----------')
        if args.tool == 'htk':
            audio_paths = path.htk(data_type='train' + data_size)
        else:
            audio_paths = path.wav(data_type='train' + data_size)

        global_mean_male, global_mean_female, global_std_male, global_std_female = read_audio(
            audio_paths=audio_paths,
            tool=args.tool,
            config=CONFIG,
            normalize=args.normalize,
            speaker_gender_dict=path.speaker_gender_dict,
            is_training=True,
            save_path=mkdir_join(input_save_path, 'train'))
        # NOTE: ex.) save_path:
        # librispeech/inputs/data_size/train/speaker/***.npy

        for data_type in ['dev_clean', 'dev_other', 'test_clean', 'test_other']:
            print('---------- %s ----------' % data_type)
            if args.tool == 'htk':
                audio_paths = path.htk(data_type=data_type)
            else:
                audio_paths = path.wav(data_type=data_type)

            read_audio(
                audio_paths=audio_paths,
                tool=args.tool,
                config=CONFIG,
                normalize=args.normalize,
                speaker_gender_dict=path.speaker_gender_dict,
                is_training=False,
                save_path=mkdir_join(input_save_path, data_type),
                global_mean_male=global_mean_male,
                global_mean_female=global_mean_female,
                global_std_male=global_std_female,
                global_std_female=global_std_female)
            # NOTE: ex.) save_path:
            # librispeech/inputs/data_size/data_type/speaker/***.npy

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'complete.txt'), 'w') as f:
            f.write('')


def make_label(data_size):

    label_save_path = mkdir_join(
        args.dataset_save_path, 'labels', data_size)

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        print('---------- train ----------')
        label_paths = path.trans(data_type='train' + data_size)
        read_trans(
            label_paths=label_paths,
            data_size=data_size,
            vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
            is_training=True,
            save_vocab_file=True,
            save_path=mkdir_join(label_save_path, 'train'))
        # NOTE: ex.) save_path:
        # librispeech/labels/data_size/train/label_type/speaker/***.npy

        for data_type in ['dev_clean', 'dev_other']:
            print('---------- %s ----------' % data_type)
            read_trans(
                label_paths=path.trans(data_type=data_type),
                data_size=data_size,
                vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
                save_path=mkdir_join(label_save_path, data_type))
            # NOTE: ex.) save_path:
            # librispeech/labels/data_size/dev_*/label_type/speaker/***.npy

        for data_type in ['test_clean', 'test_other']:
            print('---------- %s ----------' % data_type)
            read_trans(
                label_paths=path.trans(data_type=data_type),
                data_size=data_size,
                vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
                is_test=True,
                save_path=mkdir_join(label_save_path, data_type))
            # NOTE: ex.) save_path:
            # librispeech/labels/data_size/test_*/speaker/***.npy

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':

    data_sizes = ['100h']
    if bool(args.medium):
        data_sizes += ['460h']
    if bool(args.large):
        data_sizes += ['960h']

    for data_size in data_sizes:
        make_input(data_size)
        make_label(data_size)
        # TODO: add phone
