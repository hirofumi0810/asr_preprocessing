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
from glob import glob

sys.path.append('../')
from csj.prepare_path import Prepare
from csj.inputs.input_data import read_audio
from csj.labels.character import read_sdb
from utils.util import mkdir_join


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to CSJ dataset')
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
parser.add_argument('--energy', type=int, default=0, help='if 1, add the energy feature')
parser.add_argument('--delta', type=int, default=1, help='if 1, add the energy feature')
parser.add_argument('--deltadelta', type=int, default=1,
                    help='if 1, double delta features are also extracted')


def main(model, train_data_size, divide_by_space):

    print('==================================================')
    print('  model: %s' % model)
    print('  train_data_size: %s' % train_data_size)
    print('  divide_by_space: %s' % str(divide_by_space))
    print('==================================================')

    args = parser.parse_args()
    prep = Prepare(args.data_path, args.run_root_path)

    input_save_path = mkdir_join(args.dataset_save_path, 'inputs', train_data_size)
    label_save_path = mkdir_join(args.dataset_save_path, 'labels', model, train_data_size)

    ####################
    # labels
    ####################
    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')) and isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        if isfile(join(label_save_path, 'complete.txt')):
            kanji_label_save_path = None
            kana_label_save_path = None
            phone_label_save_path = None
            # NOTE: do not save
        elif divide_by_space:
            kanji_label_save_path = mkdir_join(label_save_path, 'kanji_wakachi')
            kana_label_save_path = mkdir_join(label_save_path, 'kana_wakachi')
            phone_label_save_path = mkdir_join(label_save_path, 'phone_wakachi')
        else:
            kanji_label_save_path = mkdir_join(label_save_path, 'kanji')
            kana_label_save_path = mkdir_join(label_save_path, 'kana')
            phone_label_save_path = mkdir_join(label_save_path, 'phone')

        speaker_dict_dict = {}  # dict of speaker_dict
        for data_type in ['train', 'dev', 'eval1', 'eval2', 'eval3']:

            print('---------- %s ----------' % data_type)
            save_map_file = True if train_data_size == 'train_fullset' and data_type == 'train' else False
            is_test = False if data_type in ['train', 'dev'] else True

            if data_type == 'train':
                data_type = train_data_size

            # Read target labels and save labels as npy files
            speaker_dict_dict[data_type] = read_sdb(
                label_paths=prep.trans(data_type=data_type),
                run_root_path=args.run_root_path,
                model=model,
                is_test=is_test,
                kanji_save_path=mkdir_join(kanji_label_save_path, data_type),
                kana_save_path=mkdir_join(kana_label_save_path, data_type),
                phone_save_path=mkdir_join(phone_label_save_path, data_type),
                save_map_file=save_map_file,
                divide_by_space=divide_by_space)

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
            config = {
                'feature_type': args.feature_type,
                'channels': args.channels,
                'sampling_rate': args.sampling_rate,
                'window': args.window,
                'slide': args.slide,
                'energy': bool(args.energy),
                'delta': bool(args.delta),
                'deltadelta': bool(args.deltadelta)
            }

            print('---------- train ----------')

            if args.tool == 'htk':
                audio_train_paths = [path for path in sorted(
                    glob(join(args.htk_save_path, train_data_size + '/*.htk')))]
                # NOTE: these are htk file paths
            else:
                audio_train_paths = prep.wav(data_type=train_data_size)

            # Read htk or wav files, and save input data and frame num dict
            train_global_mean_male, train_global_mean_female, train_global_std_male, train_global_std_female = read_audio(
                audio_paths=audio_train_paths,
                speaker_dict=speaker_dict_dict['train'],
                tool=args.tool,
                config=config,
                normalize=args.normalize,
                is_training=True,
                save_path=mkdir_join(input_save_path, 'train'))

            for data_type in ['dev', 'eval1', 'eval2',  'eval3']:
                print('---------- %s ----------' % data_type)

                if args.tool == 'htk':
                    audio_paths = [path for path in sorted(
                        glob(join(args.htk_save_path, data_type + '/*.htk')))]
                    # NOTE: these are htk file paths
                else:
                    audio_paths = prep.wav(data_type=data_type)

                read_audio(audio_paths=audio_paths,
                           speaker_dict=speaker_dict_dict[data_type],
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
    for model in ['ctc', 'attention']:
        for train_data_size in ['train_fullset', 'train_subset']:
            for divide_by_space in [False, True]:
                main(model, train_data_size, divide_by_space)
        # TODO: remove this loop
