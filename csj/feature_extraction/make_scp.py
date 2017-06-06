#! /usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join
import sys

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from utils.util import mkdir, mkdir_join
from utils.config.make_config import setup


def main():

    prep = Prepare()

    wav_train_paths = prep.wav(data_type='train')
    wav_train_all_paths = prep.wav(data_type='train_all')
    wav_eval1_paths = prep.wav(data_type='eval1')
    wav_eval2_paths = prep.wav(data_type='eval2')
    wav_eval3_paths = prep.wav(data_type='eval3')
    wav_dialog_train_paths = prep.wav(data_type='dialog_train')
    wav_dialog_dev_paths = prep.wav(data_type='dialog_dev')
    wav_dialog_test_paths = prep.wav(data_type='dialog_test')

    save_train_path = mkdir_join(prep.fbank_path, 'train')
    save_train_all_path = mkdir_join(prep.fbank_path, 'train_all')
    save_eval1_path = mkdir_join(prep.fbank_path, 'eval1')
    save_eval2_path = mkdir_join(prep.fbank_path, 'eval2')
    save_eval3_path = mkdir_join(prep.fbank_path, 'eval3')
    save_dialog_train_path = mkdir_join(prep.fbank_path, 'dialog_train')
    save_dialog_dev_path = mkdir_join(prep.fbank_path, 'dialog_dev')
    save_dialog_test_path = mkdir_join(prep.fbank_path, 'dialog_test')

    # HTK settings
    setup(corpus='csj',
          feature='fbank',
          dim=40,
          sampling_rate=16000,
          window=0.025,
          slide=0.01,
          energy=True,
          delta=True,
          deltadelta=True,
          window_func='hamming')

    ################
    # train (240h)
    ################
    if len(wav_train_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_train_paths) != 986:
        raise ValueError('File number is not correct (True: 986, Now: %d).' %
                         len(wav_train_paths))
    with open('wav2fbank_train.scp', 'w') as f:
        for wav_path in wav_train_paths:
            speaker_name = wav_path.split('/')[-1].split('.')[0]
            save_path = join(save_train_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ################
    # train (586h)
    ################
    if len(wav_train_all_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_train_all_paths) != 3212:
        raise ValueError('File number is not correct (True: 3212, Now: %d).' %
                         len(wav_train_all_paths))
    with open('wav2fbank_train_all.scp', 'w') as f:
        for wav_path in wav_train_all_paths:
            speaker_name = wav_path.split('/')[-1].split('.')[0]
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
    with open('wav2fbank_eval1.scp', 'w') as f:
        for wav_path in wav_eval1_paths:
            speaker_name = wav_path.split('/')[-1].split('.')[0]
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
    with open('wav2fbank_eval2.scp', 'w') as f:
        for wav_path in wav_eval2_paths:
            speaker_name = wav_path.split('/')[-1].split('.')[0]
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

    with open('wav2fbank_eval3.scp', 'w') as f:
        for wav_path in wav_eval3_paths:
            speaker_name = wav_path.split('/')[-1].split('.')[0]
            save_path = join(save_eval3_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ################
    # dialog (train)
    ################
    if len(wav_dialog_train_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_dialog_train_paths) != 100:
        raise ValueError('File number is not correct (True: 100, Now: %d).' %
                         len(wav_dialog_train_paths))
    with open('wav2fbank_dialog_train.scp', 'w') as f:
        for wav_path in wav_dialog_train_paths:
            speaker_name = wav_path.split('/')[-1].split('.')[0]
            save_path = join(save_dialog_train_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ################
    # dialog (dev)
    ################
    if len(wav_dialog_dev_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_dialog_dev_paths) != 7:
        raise ValueError('File number is not correct (True: 7, Now: %d).' %
                         len(wav_dialog_dev_paths))
    with open('wav2fbank_dialog_dev.scp', 'w') as f:
        for wav_path in wav_dialog_dev_paths:
            speaker_name = wav_path.split('/')[-1].split('.')[0]
            save_path = join(save_dialog_dev_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ################
    # dialog (test)
    ################
    if len(wav_dialog_test_paths) == 0:
        raise ValueError('There are not any wav files.')
    elif len(wav_dialog_test_paths) != 8:
        raise ValueError('File number is not correct (True: 8, Now: %d).' %
                         len(wav_dialog_test_paths))
    with open('wav2fbank_dialog_test.scp', 'w') as f:
        for wav_path in wav_dialog_test_paths:
            speaker_name = wav_path.split('/')[-1].split('.')[0]
            save_path = join(save_dialog_test_path, speaker_name + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')


if __name__ == '__main__':
    main()
