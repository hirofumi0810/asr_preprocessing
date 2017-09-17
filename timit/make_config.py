#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make configuration file for HTK toolkit (TIMIT corpus)."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os.path import join, basename
import argparse

sys.path.append('../')
from timit.prepare_path import Prepare
from utils.util import mkdir_join, mkdir
from utils.inputs.htk import save

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
parser.add_argument('--htk_save_path', type=str, help='path to save htk files')
parser.add_argument('--run_root_path', type=str, help='path to run this script')

parser.add_argument('--feature_type', type=str, default='logmelfbank',
                    help='the type of features, logmelfbank or mfcc')
parser.add_argument('--channels', type=int, default=40, help='the number of frequency channels')
parser.add_argument('--sampling_rate', type=float, default=16000, help='sampling rate')
parser.add_argument('--window', type=float, default=0.025, help='window width to extract features')
parser.add_argument('--slide', type=float, default=0.01, help='extract features per \'slide\'')
parser.add_argument('--energy', type=bool, default=True, help='if True, add the energy feature')
parser.add_argument('--delta', type=bool, default=True, help='if True, add the energy feature')
parser.add_argument('--deltadelta', type=bool, default=True,
                    help='if True, double delta features are also extracted')
parser.add_argument('--config_path', type=str, help='path to save the config file')


def main():

    args = parser.parse_args()
    htk_save_path = mkdir(args.htk_save_path)
    prep = Prepare(args.data_path, args.run_root_path)

    # HTK settings
    save(audio_file_type='nist',
         feature_type=args.feature_type,
         channels=args.channels,
         config_path=args.config_path,
         sampling_rate=args.sampling_rate,
         window=args.window,
         slide=args.slide,
         energy=args.energy,
         delta=args.delta,
         deltadelta=args.deltadelta)
    # NOTE: 120-dim features are extracted by default

    file_number = {
        'train': 3696,
        'dev': 400,
        'test': 192
    }

    for data_type in ['train', 'dev', 'test']:

        wav_paths = prep.wav(data_type=data_type)
        save_path = mkdir_join(htk_save_path, data_type)

        assert len(wav_paths) == file_number[data_type], 'File number is not correct (True: %d, Now: %d).'.format(
            file_number[data_type], len(wav_paths))

        with open(join(args.run_root_path, 'config/wav2fbank_' + data_type + '.scp'), 'w') as f:
            for wav_path in wav_paths:
                speaker_name = wav_path.split('/')[-2]
                wav_index = basename(wav_path).split('.')[0]
                save_path_tmp = join(save_path, speaker_name + '_' + wav_index + '.htk')
                f.write(wav_path + '  ' + save_path_tmp + '\n')


if __name__ == '__main__':
    main()
