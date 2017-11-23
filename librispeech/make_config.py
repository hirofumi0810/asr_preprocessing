#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os.path import join, basename
from glob import glob
import argparse

sys.path.append('../')
from utils.util import mkdir_join, mkdir
from utils.inputs.htk import save_config

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to Librispeech dataset')
parser.add_argument('--htk_save_path', type=str, help='path to save htk files')

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
parser.add_argument('--medium', type=int,
                    help='If True, create medium-size dataset (460h).')
parser.add_argument('--large', type=int,
                    help='If True, create large-size dataset (960h).')


def main():

    args = parser.parse_args()
    htk_save_path = mkdir(args.htk_save_path)

    # HTK settings
    save_config(audio_file_type='wav',
                feature_type=args.feature_type,
                channels=args.channels,
                config_save_path='./config',
                sampling_rate=16000,
                window=args.window,
                slide=args.slide,
                energy=bool(args.energy),
                delta=bool(args.delta),
                deltadelta=bool(args.deltadelta))
    # NOTE: 120-dim features are extracted by default

    parts = ['train-clean-100', 'dev-clean', 'dev-other',
             'test-clean', 'test-other']

    if bool(args.large):
        parts += ['train-clean-360', 'train-other-500']
    elif bool(args.medium):
        parts += ['train-clean-360']

    for part in parts:
        # part/speaker/book/*.wav
        wav_paths = [p for p in glob(join(args.data_path, part, '*/*/*.wav'))]
        with open('./config/wav2htk_' + part + '.scp', 'w') as f:
            for wav_path in wav_paths:
                # ex.) wav_path: speaker/book/speaker-book-utt_index.wav
                speaker, book, utt_index = basename(
                    wav_path).split('.')[0].split('-')
                save_path = mkdir_join(
                    htk_save_path, part, speaker, book, basename(wav_path).split('.')[0] + '.htk')
                f.write(wav_path + '  ' + save_path + '\n')
                # ex.) htk_path: speaker/book/speaker-book-utt_index.htk


if __name__ == '__main__':
    main()
