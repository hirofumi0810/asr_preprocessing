#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import argparse
from glob import glob

sys.path.append('../')
from prepare_path import Prepare
from inputs.input_data import read_wav
from utils.util import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
parser.add_argument('--dataset_save_path', type=str, help='path to save dataset')
parser.add_argument('--run_root_path', type=str, help='path to run this script')
parser.add_argument('--tool', type=str,
                    help='the tool to extract features, htk or python_speech_features or htk')
parser.add_argument('--htk_save_path', type=str, default='',
                    help='path to save features, this is needed only when you use HTK.')
parser.add_argument('--normalize', type=str, default='speaker',
                    help='global or speaker or utterance')

parser.add_argument('--feature_type', type=str, default='logmelfbank',
                    help='the type of features, logmelfbank or mfcc or linearmelfbank')
parser.add_argument('--channels', type=int, default=40, help='the number of frequency channels')
parser.add_argument('--sampling_rate', type=float, default=16000, help='sampling rate')
parser.add_argument('--window', type=float, default=0.025, help='window width to extract features')
parser.add_argument('--slide', type=float, default=0.01, help='extract features per \'slide\'')
parser.add_argument('--energy', type=bool, default=True, help='if True, add the energy feature')
parser.add_argument('--delta', type=bool, default=True, help='if True, add the energy feature')
parser.add_argument('--deltadelta', type=bool, default=True,
                    help='if True, double delta features are also extracted')


def main():

    args = parser.parse_args()
    prep = Prepare(args.data_path, args.run_root_path)
    if args.tool == 'htk':
        wav_train_paths = [path for path in glob(join(args.htk_save_path, 'train/*.htk'))]
        wav_dev_paths = [path for path in glob(join(args.htk_save_path, 'dev/*.htk'))]
        wav_test_paths = [path for path in glob(join(args.htk_save_path, 'test/*.htk'))]
        # NOTE: these are htk file paths
    else:
        wav_train_paths = prep.wav(data_type='train')
        wav_dev_paths = prep.wav(data_type='dev')
        wav_test_paths = prep.wav(data_type='test')
    input_save_path = mkdir_join(args.dataset_save_path, args.tool, args.normalize, 'inputs')

    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        input_train_save_path = mkdir_join(input_save_path, 'train')
        input_dev_save_path = mkdir_join(input_save_path, 'dev')
        input_test_save_path = mkdir_join(input_save_path, 'test')

        config = {
            'feature_type': args.feature_type,
            'channels': args.channels,
            'sampling_rate': args.sampling_rate,
            'window': args.window,
            'slide': args.slide,
            'energy': args.energy,
            'delta': args.delta,
            'deltadelta': args.deltadelta
        }

        print('=> Processing input data...')
        print('---------- train ----------')
        train_global_mean, train_global_std = read_wav(
            wav_paths=wav_train_paths,
            tool=args.tool,
            config=config,
            save_path=input_train_save_path,
            normalize=args.normalize,
            is_training=True)

        print('---------- dev ----------')
        read_wav(wav_paths=wav_dev_paths,
                 tool=args.tool,
                 config=config,
                 save_path=input_dev_save_path,
                 normalize=args.normalize,
                 is_training=False,
                 train_global_mean=train_global_mean,
                 train_global_std=train_global_std)

        print('---------- test ----------')
        read_wav(wav_paths=wav_test_paths,
                 tool=args.tool,
                 config=config,
                 save_path=input_test_save_path,
                 normalize=args.normalize,
                 is_training=False,
                 train_global_mean=train_global_mean,
                 train_global_std=train_global_std)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':
    main()
