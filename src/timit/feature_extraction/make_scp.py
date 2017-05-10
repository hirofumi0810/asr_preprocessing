#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from utils.util import mkdir
from utils.config.make_config import setup


def main():

    prep = Prepare()

    wav_train_paths = prep.wav(label_type='train')
    wav_dev_paths = prep.wav(label_type='dev')
    wav_test_paths = prep.wav(label_type='test')

    save_path = mkdir(os.path.join(prep.root_path, 'fbank'))
    save_path_train = mkdir(os.path.join(save_path, 'train'))
    save_path_dev = mkdir(os.path.join(save_path, 'dev'))
    save_path_test = mkdir(os.path.join(save_path, 'test'))

    # HTK settings
    setup(corpus='timit', feature='fbank', dim=40, sampling_rate=16000, window=0.025, slide=0.01,
          energy=True, delta=True, deltadelta=True, window_func='hamming')

    ####################
    # train
    ####################
    if len(wav_train_paths) == 0:
        raise ValueError('Error: Convert from sph to wav files.')
    with open('wav2fbank_train.scp', 'w') as f:
        for wav_train_path in sorted(wav_train_paths):
            speaker_name = wav_train_path.split('/')[-2]
            wav_index = os.path.basename(wav_train_path).split('.')[0]
            save_path = os.path.join(save_path_train, speaker_name + '_' + wav_index + '.htk')
            f.write(wav_train_path + '  ' + save_path + '\n')
    if len(wav_train_paths) != 3696:
        raise ValueError('Error: File number is not correct (True: 3696, Now: %d).' %
                         len(wav_train_paths))

    ####################
    # dev
    ####################
    if len(wav_dev_paths) == 0:
        raise ValueError('Error: Convert from sph to wav files.')
    with open('wav2fbank_dev.scp', 'w') as f:
        for wav_dev_path in sorted(wav_dev_paths):
            speaker_name = wav_dev_path.split('/')[-2]
            wav_index = os.path.basename(wav_dev_path).split('.')[0]
            save_path = os.path.join(save_path_dev, speaker_name + '_' + wav_index + '.htk')
            f.write(wav_dev_path + '  ' + save_path + '\n')
    if len(wav_dev_paths) != 400:
        raise ValueError('Error: File number is not correct (True: 400, Now: %d).' %
                         len(wav_dev_paths))

    ####################
    # test
    ####################
    if len(wav_test_paths) == 0:
        raise ValueError('Error: Convert from sph to wav files.')
    with open('wav2fbank_test.scp', 'w') as f:
        for wav_test_path in sorted(wav_test_paths):
            speaker_name = wav_test_path.split('/')[-2]
            wav_index = os.path.basename(wav_test_path).split('.')[0]
            save_path = os.path.join(save_path_test, speaker_name + '_' + wav_index + '.htk')
            f.write(wav_test_path + '  ' + save_path + '\n')
    if len(wav_test_paths) != 192:
        raise ValueError('Error: File number is not correct (True: 192, Now: %d).' %
                         len(wav_test_paths))


if __name__ == '__main__':
    main()
