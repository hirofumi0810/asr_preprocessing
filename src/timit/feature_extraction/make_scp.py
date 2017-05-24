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

    wav_train_paths = prep.wav(data_type='train')
    wav_dev_paths = prep.wav(data_type='dev')
    wav_test_paths = prep.wav(data_type='test')

    save_train_path = mkdir(os.path.join(prep.fbank_path, 'train'))
    save_dev_path = mkdir(os.path.join(prep.fbank_path, 'dev'))
    save_test_path = mkdir(os.path.join(prep.fbank_path, 'test'))

    # HTK settings
    setup(corpus='timit',
          feature='fbank',
          dim=40,
          sampling_rate=16000,
          window=0.025,
          slide=0.01,
          energy=True,
          delta=True,
          deltadelta=True,
          window_func='hamming')

    ####################
    # train
    ####################
    if len(wav_train_paths) == 0:
        raise ValueError('There is no wav file.')
    elif len(wav_train_paths) != 3696:
        raise ValueError('File number is not correct (True: 3696, Now: %d).' %
                         len(wav_train_paths))
    with open('wav2fbank_train.scp', 'w') as f:
        for wav_path in wav_train_paths:
            speaker_name = wav_path.split('/')[-2]
            wav_index = os.path.basename(wav_path).split('.')[0]
            save_path = os.path.join(
                save_train_path, speaker_name + '_' + wav_index + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ####################
    # dev
    ####################
    if len(wav_dev_paths) == 0:
        raise ValueError('There is no wav file.')
    elif len(wav_dev_paths) != 400:
        raise ValueError('File number is not correct (True: 400, Now: %d).' %
                         len(wav_dev_paths))
    with open('wav2fbank_dev.scp', 'w') as f:
        for wav_path in wav_dev_paths:
            speaker_name = wav_path.split('/')[-2]
            wav_index = os.path.basename(wav_path).split('.')[0]
            save_path = os.path.join(
                save_dev_path, speaker_name + '_' + wav_index + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ####################
    # test
    ####################
    if len(wav_test_paths) == 0:
        raise ValueError('There is no wav file.')
    elif len(wav_test_paths) != 192:
        raise ValueError('File number is not correct (True: 192, Now: %d).' %
                         len(wav_test_paths))
    with open('wav2fbank_test.scp', 'w') as f:
        for wav_path in wav_test_paths:
            speaker_name = wav_path.split('/')[-2]
            wav_index = os.path.basename(wav_path).split('.')[0]
            save_path = os.path.join(
                save_test_path, speaker_name + '_' + wav_index + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')


if __name__ == '__main__':
    main()
