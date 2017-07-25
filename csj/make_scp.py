#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, basename
import sys

sys.path.append('../')
from csj.prepare_path import Prepare
from utils.util import mkdir_join, mkdir
from utils.htk.make_config import setup


def main(data_path, input_feature_save_path, run_root_path):

    prep = Prepare(data_path, run_root_path)
    wav_train_subset_paths = prep.wav(data_type='train_subset')
    wav_train_fullset_paths = prep.wav(data_type='train_fullset')
    wav_eval1_paths = prep.wav(data_type='eval1')
    wav_eval2_paths = prep.wav(data_type='eval2')
    wav_eval3_paths = prep.wav(data_type='eval3')
    wav_dialog_paths = prep.wav(data_type='dialog')

    save_train_path = mkdir_join(input_feature_save_path, 'train_subset')
    save_train_all_path = mkdir_join(input_feature_save_path, 'train_fullset')
    save_eval1_path = mkdir_join(input_feature_save_path, 'eval1')
    save_eval2_path = mkdir_join(input_feature_save_path, 'eval2')
    save_eval3_path = mkdir_join(input_feature_save_path, 'eval3')
    save_dialog_path = mkdir_join(input_feature_save_path, 'dialog')

    # HTK settings
    setup(audio_file_type='wav',
          feature='fbank',
          channels=40,
          save_path=abspath('./config'),
          sampling_rate=16000,
          window=0.025,
          slide=0.01,
          energy=False,
          delta=True,
          deltadelta=True,
          window_func='hamming')

    ################
    # train_subset (240h)
    ################
    if len(wav_train_subset_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_train_subset_paths) != 986:
        raise ValueError('File number is not correct (True: 986, Now: %d).' %
                         len(wav_train_subset_paths))
    with open(join(run_root_path, 'config/wav2fbank_train_subset.scp'), 'w') as f:
        for wav_path in wav_train_subset_paths:
            speaker_name = basename(wav_path).split('.')[0]
            save_path = join(save_train_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    #####################
    # train_fullset (586h)
    #####################
    if len(wav_train_fullset_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_train_fullset_paths) != 3212:
        raise ValueError('File number is not correct (True: 3212, Now: %d).' %
                         len(wav_train_fullset_paths))
    with open(join(run_root_path, 'config/wav2fbank_train_fullset.scp'), 'w') as f:
        for wav_path in wav_train_fullset_paths:
            speaker_name = basename(wav_path).split('.')[0]
            save_path = join(save_train_all_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ################
    # eval1
    ################
    if len(wav_eval1_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_eval1_paths) != 10:
        raise ValueError('File number is not correct (True: 10, Now: %d).' %
                         len(wav_eval1_paths))
    with open(join(run_root_path, 'config/wav2fbank_eval1.scp'), 'w') as f:
        for wav_path in wav_eval1_paths:
            speaker_name = basename(wav_path).split('.')[0]
            save_path = join(save_eval1_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ################
    # eval2
    ################
    if len(wav_eval2_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_eval2_paths) != 10:
        raise ValueError('File number is not correct (True: 10, Now: %d).' %
                         len(wav_eval2_paths))
    with open(join(run_root_path, 'config/wav2fbank_eval2.scp'), 'w') as f:
        for wav_path in wav_eval2_paths:
            speaker_name = basename(wav_path).split('.')[0]
            save_path = join(save_eval2_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ################
    # eval3
    ################
    if len(wav_eval3_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_eval3_paths) != 10:
        raise ValueError('File number is not correct (True: 10, Now: %d).' %
                         len(wav_eval3_paths))
    with open(join(run_root_path, 'config/wav2fbank_eval3.scp'), 'w') as f:
        for wav_path in wav_eval3_paths:
            speaker_name = basename(wav_path).split('.')[0]
            save_path = join(save_eval3_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ################
    # dialog
    ################
    if len(wav_dialog_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_dialog_paths) != 15:
        raise ValueError('File number is not correct (True: 10, Now: %d).' %
                         len(wav_dialog_paths))
    with open(join(run_root_path, 'config/wav2fbank_dialog.scp'), 'w') as f:
        for wav_path in wav_dialog_paths:
            speaker_name = basename(wav_path).split('.')[0]
            save_path = join(save_dialog_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 4:
        raise ValueError

    data_path = args[1]
    input_feature_save_path = mkdir(args[2])
    run_root_path = args[3]

    main(data_path, input_feature_save_path, run_root_path)
