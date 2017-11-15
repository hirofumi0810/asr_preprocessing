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

sys.path.append('../')
from csj.path import Path
from csj.input_data import read_audio
from csj.labels.transcript import read_sdb
from utils.util import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to CSJ dataset')
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
parser.add_argument('--energy', type=int, default=0,
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

    print('=' * 30)
    print('  data_size: %s' % data_size)
    print('=' * 30)

    input_save_path = mkdir_join(
        args.dataset_save_path, 'inputs', data_size)
    label_save_path = mkdir_join(
        args.dataset_save_path, 'labels', data_size)

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')) and isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        ####################
        # labels
        ####################
        speaker_dict_dict = {}  # dict of speaker_dict

        print('---------- train ----------')
        speaker_dict_dict['train'] = read_sdb(
            label_paths=path.trans(data_type='train_' + data_size),
            data_size=data_size,
            vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
            is_training=True,
            save_vocab_file=True,
            save_path=mkdir_join(label_save_path, 'train'))
        # NOTE: ex.) save_path:
        # csj/labels/data_size/train/label_type/speaker/*.npy

        for data_type in ['dev', 'eval1', 'eval2', 'eval3']:
            print('---------- %s ----------' % data_type)
            is_test = False if data_type == 'dev' else True

            speaker_dict_dict[data_type] = read_sdb(
                label_paths=path.trans(data_type=data_type),
                data_size=data_size,
                vocab_file_save_path=mkdir_join('./config', 'vocab_files'),
                is_test=is_test,
                save_path=mkdir_join(label_save_path, data_type))
            # NOTE: ex.) save_path:
            # csj/labels/data_size/data_type/label_type/speaker/*.npy

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')

        ####################
        # inputs
        ####################
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
            # csj/inputs/data_size/train/speaker/*.npy

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
                # csj/inputs/data_size/data_type/speaker/*.npy

            # Make a confirmation file to prove that dataset was saved
            # correctly
            with open(join(input_save_path, 'complete.txt'), 'w') as f:
                f.write('')


if __name__ == '__main__':

    data_sizes = []
    if bool(args.subset):
        data_sizes += ['subset']
    if bool(args.fullset):
        data_sizes += ['fullset']

    for data_size in data_sizes:
        main(data_size)
