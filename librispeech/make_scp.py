#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, basename
import sys

sys.path.append('../')
from librispeech.prepare_path import Prepare
from utils.util import mkdir_join, mkdir
from utils.htk.make_config import setup


def main(data_path, input_feature_save_path, run_root_path):

    prep = Prepare(data_path, run_root_path)
    wav_train_clean100_paths = prep.wav(data_type='train_clean100')
    wav_train_clean360_paths = prep.wav(data_type='train_clean360')
    wav_train_other500_paths = prep.wav(data_type='train_other500')
    wav_dev_clean_paths = prep.wav(data_type='dev_clean')
    wav_dev_other_paths = prep.wav(data_type='dev_other')
    wav_test_clean_paths = prep.wav(data_type='test_clean')
    wav_test_other_paths = prep.wav(data_type='test_other')

    save_train_clean100_path = mkdir_join(
        input_feature_save_path, 'train_clean100')
    save_train_clean360_path = mkdir_join(
        input_feature_save_path, 'train_clean360')
    save_train_other500_path = mkdir_join(
        input_feature_save_path, 'train_other500')
    save_dev_clean_path = mkdir_join(input_feature_save_path, 'dev_clean')
    save_dev_other_path = mkdir_join(input_feature_save_path, 'dev_other')
    save_test_clean_path = mkdir_join(input_feature_save_path, 'test_clean')
    save_test_other_path = mkdir_join(input_feature_save_path, 'test_other')

    # HTK settings
    setup(audio_file_type='wav',
          feature='fbank',
          channels=40,
          save_path=abspath('./config'),
          sampling_rate=16000,
          window=0.025,
          slide=0.01,
          energy=True,
          delta=True,
          deltadelta=True,
          window_func='hamming')

    #######################
    # train (clean, 100h)
    #######################
    if len(wav_train_clean100_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_train_clean100_paths) != 28539:
        raise ValueError('File number is not correct (True: 28539, Now: %d).' %
                         len(wav_train_clean100_paths))
    with open(join(run_root_path,
                   'config/wav2fbank_train_clean100.scp'), 'w') as f:
        for wav_path in wav_train_clean100_paths:
            uttrance_name = basename(wav_path).split('.')[0]
            speaker_index, book_index, uttrance_index = uttrance_name.split(
                '-')
            mkdir_join(save_train_clean100_path, speaker_index)
            save_path = join(save_train_clean100_path, speaker_index,
                             uttrance_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    #######################
    # train (clean, 360h)
    #######################
    if len(wav_train_clean360_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_train_clean360_paths) != 104014:
        raise ValueError('File number is not correct (True: 104014, Now: %d).' %
                         len(wav_train_clean360_paths))
    with open(join(run_root_path,
                   'config/wav2fbank_train_clean360.scp'), 'w') as f:
        for wav_path in wav_train_clean360_paths:
            uttrance_name = basename(wav_path).split('.')[0]
            speaker_index, book_index, uttrance_index = uttrance_name.split(
                '-')
            mkdir_join(save_train_clean360_path, speaker_index)
            save_path = join(save_train_clean360_path, speaker_index,
                             uttrance_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    #######################
    # train (other, 500h)
    #######################
    if len(wav_train_other500_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_train_other500_paths) != 148688:
        raise ValueError('File number is not correct (True: 148688, Now: %d).' %
                         len(wav_train_other500_paths))
    with open(join(run_root_path,
                   'config/wav2fbank_train_other500.scp'), 'w') as f:
        for wav_path in wav_train_other500_paths:
            uttrance_name = basename(wav_path).split('.')[0]
            speaker_index, book_index, uttrance_index = uttrance_name.split(
                '-')
            mkdir_join(save_train_other500_path, speaker_index)
            save_path = join(save_train_other500_path, speaker_index,
                             uttrance_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    #######################
    # dev (clean)
    #######################
    if len(wav_dev_clean_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_dev_clean_paths) != 2703:
        raise ValueError('File number is not correct (True: 2703, Now: %d).' %
                         len(wav_dev_clean_paths))
    with open(join(run_root_path,
                   'config/wav2fbank_dev_clean.scp'), 'w') as f:
        for wav_path in wav_dev_clean_paths:
            uttrance_name = basename(wav_path).split('.')[0]
            speaker_index, book_index, uttrance_index = uttrance_name.split(
                '-')
            mkdir_join(save_dev_clean_path, speaker_index)
            save_path = join(save_dev_clean_path, speaker_index,
                             uttrance_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    #######################
    # dev (other)
    #######################
    if len(wav_dev_other_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_dev_other_paths) != 2864:
        raise ValueError('File number is not correct (True: 2864, Now: %d).' %
                         len(wav_dev_other_paths))
    with open(join(run_root_path,
                   'config/wav2fbank_dev_other.scp'), 'w') as f:
        for wav_path in wav_dev_other_paths:
            uttrance_name = basename(wav_path).split('.')[0]
            speaker_index, book_index, uttrance_index = uttrance_name.split(
                '-')
            mkdir_join(save_dev_other_path, speaker_index)
            save_path = join(save_dev_other_path, speaker_index,
                             uttrance_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    #######################
    # test (clean)
    #######################
    if len(wav_test_clean_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_test_clean_paths) != 2620:
        raise ValueError('File number is not correct (True: 2620, Now: %d).' %
                         len(wav_test_clean_paths))
    with open(join(run_root_path,
                   'config/wav2fbank_test_clean.scp'), 'w') as f:
        for wav_path in wav_test_clean_paths:
            uttrance_name = basename(wav_path).split('.')[0]
            speaker_index, book_index, uttrance_index = uttrance_name.split(
                '-')
            mkdir_join(save_test_clean_path, speaker_index)
            save_path = join(save_test_clean_path, speaker_index,
                             uttrance_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    #######################
    # test (other)
    #######################
    if len(wav_test_other_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_test_other_paths) != 2939:
        raise ValueError('File number is not correct (True: 2939, Now: %d).' %
                         len(wav_test_other_paths))
    with open(join(run_root_path,
                   'config/wav2fbank_test_other.scp'), 'w') as f:
        for wav_path in wav_test_other_paths:
            uttrance_name = basename(wav_path).split('.')[0]
            speaker_index, book_index, uttrance_index = uttrance_name.split(
                '-')
            mkdir_join(save_test_other_path, speaker_index)
            save_path = join(save_test_other_path, speaker_index,
                             uttrance_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 4:
        raise ValueError

    data_path = args[1]
    input_feature_save_path = mkdir(args[2])
    run_root_path = args[3]

    main(data_path, input_feature_save_path, run_root_path)
