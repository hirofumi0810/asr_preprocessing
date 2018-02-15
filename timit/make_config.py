#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make configuration file for HTK toolkit (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import sys
import argparse

sys.path.append('../')
from timit.path import Path
from utils.util import mkdir_join, mkdir
from utils.inputs.htk import save_config

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
parser.add_argument('--htk_save_path', type=str, help='path to save htk files')
parser.add_argument('--config_path', type=str, help='path to config directory')

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


def main():

    args = parser.parse_args()
    htk_save_path = mkdir(args.htk_save_path)
    path = Path(data_path=args.data_path,
                config_path=args.config_path)

    # HTK settings
    save_config(audio_file_type='nist',
                feature_type=args.feature_type,
                channels=args.channels,
                config_save_path='./config',
                sampling_rate=16000,
                window=args.window,
                slide=args.slide,
                energy=bool(args.energy),
                delta=bool(args.delta),
                deltadelta=bool(args.deltadelta))
    # NOTE: 123-dim features are extracted by default

    for data_type in ['train', 'dev', 'test']:

        wav_paths = path.wav(data_type=data_type)
        save_path = mkdir_join(htk_save_path, data_type)

        with open('./config/wav2htk_' + data_type + '.scp', 'w') as f:
            for wav_path in wav_paths:
                speaker = wav_path.split('/')[-2]
                utt_index = basename(wav_path).split('.')[0]
                save_path_tmp = mkdir_join(
                    save_path, speaker, utt_index + '.htk')
                f.write(wav_path + '  ' + save_path_tmp + '\n')


if __name__ == '__main__':
    main()
