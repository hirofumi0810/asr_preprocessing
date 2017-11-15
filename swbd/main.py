#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import argparse

sys.path.append('../')
from swbd.path import Path
from swbd.inputs.input_data import read_audio
from swbd.labels.ldc97s62.character import read_trans
from swbd.labels.eval2000.swbd_ch_stm import read_stm
from utils.util import mkdir_join


parser = argparse.ArgumentParser()
parser.add_argument('--swbd_audio_path', type=str,
                    help='path to LDC97S62 audio files')
parser.add_argument('--swbd_trans_path', type=str,
                    help='path to LDC97S62 transciption files')
parser.add_argument('--fisher_path', type=str, default=None,
                    help='path to Fisher dataset')
parser.add_argument('--eval2000_audio_path', type=str,
                    help='path to audio files of eval2000 dataset')
parser.add_argument('--eval2000_trans_path', type=str,
                    help='path to transcript files of eval2000 dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--run_root_path', type=str,
                    help='path to run this script')
parser.add_argument('--tool', type=str,
                    help='the tool to extract features, htk or python_speech_features or htk')
parser.add_argument('--wav_save_path', type=str, default='',
                    help='path to wav files. This is needed only when you do not use HTK.')
parser.add_argument('--htk_save_path', type=str, default='',
                    help='path to htk files. This is needed only when you use HTK.')
parser.add_argument('--normalize', type=str, default='speaker',
                    help='global or speaker or utterance')

parser.add_argument('--feature_type', type=str, default='logmelfbank',
                    help='the type of features, logmelfbank or mfcc or linearmelfbank')
parser.add_argument('--channels', type=int, default=40,
                    help='the number of frequency channels')
parser.add_argument('--sampling_rate', type=float,
                    default=8000, help='sampling rate')
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

args = parser.parse_args()
path = Path(
    swbd_audio_path=args.swbd_audio_path,
    swbd_trans_path=args.swbd_trans_path,
    fisher_path=args.fisher_path,
    eval2000_audio_path=args.eval2000_audio_path,
    eval2000_trans_path=args.eval2000_trans_path,
    wav_save_path=args.wav_save_path,
    htk_save_path=args.htk_save_path,
    run_root_path='./')

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

    label_save_path = mkdir_join(
        args.dataset_save_path, 'labels', data_size)
    # swbd/dataset/labels/data_size/model/label_type/data_type/speaker/*.npy

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        print('---------- train ----------')
        speaker_dict_train = read_trans(
            label_paths=path.trans(corpus='swbd'),
            run_root_path='./',
            save_path=mkdir_join(label_save_path, 'train'))
        # NOTE: ex.) save_path:
        # swbd/dataset/labels/data_size/train/label_type/speaker/*.npy

        print('---------- eval2000 (swbd, ch) ----------')
        # speaker_dict_test_swbd, speaker_dict_test_ch = read_stm(
        #     stm_path=path.stm_path,
        #     pem_path=path.pem_path,
        #     run_root_path='./',
        #     save_path=label_save_path)

        # Make a confirmation file to prove that dataset was saved correctly
        # with open(join(label_save_path, 'complete.txt'), 'w') as f:
        #     f.write('')

    input_save_path = mkdir_join(
        args.dataset_save_path, 'inputs', data_size)

    print('=> Processing input data...')
    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        print('---------- train ----------')
        if args.tool == 'htk':
            audio_paths = path.htk(corpus='swbd')
        else:
            audio_paths = path.wav(corpus='swbd')

        global_mean, global_std = read_audio(
            audio_paths=audio_paths,
            tool=args.tool,
            config=CONFIG,
            normalize=args.normalize,
            speaker_dict=speaker_dict_train,
            is_training=True,
            save_path=mkdir_join(input_save_path, 'train'))
        # NOTE: ex.) save_path:
        # swbd/dataset/inputs/data_size/train/speaker/*.npy

        print('---------- eval2000 (swbd) ----------')
        if args.tool == 'htk':
            audio_paths = path.htk(corpus='eval2000_swbd')
        else:
            audio_paths = path.wav(corpus='eval2000_swbd')

        # read_audio(audio_paths=audio_paths,
        #            tool=args.tool,
        #            config=CONFIG,
        #            normalize=args.normalize,
        #            speaker_gender_dict=path.speaker_gender_dict,
        #            is_training=False,
        #            save_path=mkdir_join(input_save_path, 'test_swbd'),
        #            global_mean=global_mean,
        #            global_std=global_std)

        print('---------- eval2000 (ch) ----------')
        if args.tool == 'htk':
            audio_paths = path.htk(corpus='eval2000_ch')
        else:
            audio_paths = path.wav(corpus='eval2000_ch')

        # Read htk or wav files, and save input data and frame num dict
        # read_audio(audio_paths=audio_paths,
        #            tool=args.tool,
        #            config=CONFIG,
        #            normalize=args.normalize,
        #            speaker_gender_dict=path.speaker_gender_dict,
        #            is_training=False,
        #            save_path=mkdir_join(input_save_path, 'test_ch'),
        #            global_mean=global_mean,
        #            global_std=global_std)

        # Make a confirmation file to prove that dataset was saved correctly
        # with open(join(input_save_path, 'complete.txt'), 'w') as f:
        #     f.write('')


if __name__ == '__main__':

    for data_size in ['300h']:
        # for data_size in ['300h', '2000h']:
        main(data_size)
