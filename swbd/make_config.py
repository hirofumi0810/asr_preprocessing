#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os.path import join, basename, isdir
import argparse
from glob import glob

sys.path.append('../')
from utils.util import mkdir_join, mkdir
from utils.inputs.htk import save

parser = argparse.ArgumentParser()
parser.add_argument('--wav_save_path', type=str,
                    help='path to audio files')
parser.add_argument('--htk_save_path', type=str, help='path to save htk files')
parser.add_argument('--run_root_path', type=str,
                    help='path to run this script')

parser.add_argument('--feature_type', type=str, default='logmelfbank',
                    help='the type of features, logmelfbank or mfcc')
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
parser.add_argument('--config_path', type=str,
                    help='path to save the config file')


def main():

    args = parser.parse_args()
    htk_save_path = mkdir(args.htk_save_path)

    # HTK settings
    save(audio_file_type='wav',
         feature_type=args.feature_type,
         channels=args.channels,
         config_save_path=args.config_path,
         sampling_rate=args.sampling_rate,
         window=args.window,
         slide=args.slide,
         energy=bool(args.energy),
         delta=bool(args.delta),
         deltadelta=bool(args.deltadelta))
    # NOTE: 120-dim features are extracted by default

    # Switchboard
    with open('./config/wav2htk_swbd.scp', 'w') as f:
        for wav_path in glob(join(args.wav_save_path, 'swbd/*.wav')):
            # ex.) wav_path: wav/swbd/***.wav
            save_path = mkdir_join(
                htk_save_path, 'swbd', basename(wav_path).split('.')[0] + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')
            # ex.) htk_path: wav/swbd/***.htk

    # eval2000 (swbd)
    with open('./config/wav2htk_eval2000_swbd.scp', 'w') as f:
        for wav_path in glob(join(args.wav_save_path, 'eval2000/swbd/*.wav')):
            # ex.) wav_path: wav/eval2000_swbd/***.wav
            save_path = mkdir_join(
                htk_save_path, 'eval2000', 'swbd', basename(wav_path).split('.')[0] + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')
            # ex.) htk_path: wav/eval2000/swbd/***.htk

    # eval2000 (callhome)
    with open('./config/wav2htk_eval2000_ch.scp', 'w') as f:
        for wav_path in glob(join(args.wav_save_path, 'eval2000/callhome/*.wav')):
            # ex.) wav_path: wav/eval2000_ch/***.wav
            save_path = mkdir_join(
                htk_save_path, 'eval2000', 'callhome', basename(wav_path).split('.')[0] + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')
            # ex.) htk_path: wav/eval2000/callhome/***.htk

    # Fisher
    if isdir(join(args.wav_save_path, 'fisher')):
        with open('./config/wav2htk_fisher.scp', 'w') as f:
            for wav_path in glob(join(args.wav_save_path, 'fisher/*/*.wav')):
                # ex.) wav_path: wav/fisher/speaker/***.wav
                speaker = wav_path.split('/')[-2]
                save_path = mkdir_join(
                    htk_save_path, 'fisher', speaker, basename(wav_path).split('.')[0] + '.htk')
                f.write(wav_path + '  ' + save_path + '\n')
                # ex.) htk_path: wav/fisher/speaker/***.htk


if __name__ == '__main__':
    main()
