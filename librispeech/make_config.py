#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os.path import join, basename
import argparse

sys.path.append('../')
from librispeech.prepare_path import Prepare
from utils.util import mkdir_join, mkdir
from utils.inputs.htk import save

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to  Librispeech dataset')
parser.add_argument('--htk_save_path', type=str, help='path to save htk files')
parser.add_argument('--run_root_path', type=str, help='path to run this script')

parser.add_argument('--feature_type', type=str, default='logmelfbank',
                    help='the type of features, logmelfbank or mfcc')
parser.add_argument('--channels', type=int, default=40, help='the number of frequency channels')
parser.add_argument('--sampling_rate', type=float, default=16000, help='sampling rate')
parser.add_argument('--window', type=float, default=0.025, help='window width to extract features')
parser.add_argument('--slide', type=float, default=0.01, help='extract features per \'slide\'')
parser.add_argument('--energy', type=int, default=0, help='if 1, add the energy feature')
parser.add_argument('--delta', type=int, default=1, help='if 1, add the energy feature')
parser.add_argument('--deltadelta', type=int, default=1,
                    help='if 1, double delta features are also extracted')
parser.add_argument('--config_path', type=str, help='path to save the config file')


def main():

    args = parser.parse_args()
    htk_save_path = mkdir(args.htk_save_path)
    prep = Prepare(args.data_path, args.run_root_path)

    # HTK settings
    save(audio_file_type='wav',
         feature_type=args.feature_type,
         channels=args.channels,
         config_path=args.config_path,
         sampling_rate=args.sampling_rate,
         window=args.window,
         slide=args.slide,
         energy=bool(args.energy),
         delta=bool(args.delta),
         deltadelta=bool(args.deltadelta))
    # NOTE: 120-dim features are extracted by default

    for data_type in ['train_clean100', 'train_clean360', 'train_other500',
                      'dev_clean', 'dev_other', 'test_clean', 'test_other']:

        wav_paths = prep.wav(data_type=data_type)
        save_path = mkdir_join(htk_save_path, data_type)

        with open(join(args.run_root_path, 'config/wav2fbank_' + data_type + '.scp'), 'w') as f:
            for wav_path in wav_paths:
                uttrance_name = basename(wav_path).split('.')[0]
                speaker_index, book_index, uttrance_index = uttrance_name.split('-')
                mkdir_join(save_path, speaker_index)
                save_path = join(
                    save_path, speaker_index, uttrance_name + '.htk')
                f.write(wav_path + '  ' + save_path + '\n')


if __name__ == '__main__':
    main()
