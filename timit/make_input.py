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
from timit.prepare_path import Prepare
from timit.inputs.input_data import read_wav
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

    input_save_path = mkdir_join(args.dataset_save_path, 'inputs', args.tool, args.normalize)

    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.\n')
    else:
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

        if args.tool == 'htk':
            wav_train_paths = [path for path in glob(join(args.htk_save_path, 'train/*.htk'))]
            # NOTE: these are htk file paths
        else:
            wav_train_paths = prep.wav(data_type='train')

        train_global_mean_male, train_global_std_male, train_global_mean_female, train_global_std_female = read_wav(
            wav_paths=wav_train_paths,
            tool=args.tool,
            config=config,
            normalize=args.normalize,
            is_training=True,
            save_path=mkdir_join(input_save_path, 'train'))

        for data_type in ['dev', 'test']:
            print('---------- %s ----------' % data_type)

            if args.tool == 'htk':
                wav_paths = [path for path in glob(join(args.htk_save_path, data_type + '/*.htk'))]
                # NOTE: these are htk file paths
            else:
                wav_paths = prep.wav(data_type=data_type)

            read_wav(wav_paths=wav_paths,
                     tool=args.tool,
                     config=config,
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


if __name__ == '__main__':
    main()
