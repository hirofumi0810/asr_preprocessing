#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from utils.util import mkdir
from utils.config.make_config import setup


def main():

    prep = Prepare()

    wav_train_path = os.path.join(prep.train_data_path, 'wav')
    wav_train_fisher_path = os.path.join(prep.train_data_fisher_path, 'wav')
    wav_test_path = os.path.join(prep.test_data_path, 'wav/swbd')
    wav_test_callhome_path = os.path.join(prep.test_data_path, 'wav/callhome')

    save_train_path = mkdir(os.path.join(prep.train_data_path, 'fbank'))
    save_train_fisher_path = mkdir(os.path.join(prep.train_data_fisher_path, 'fbank'))
    save_test_path = mkdir(os.path.join(prep.test_data_path, 'fbank/swbd'))
    save_test_callhome_path = mkdir(os.path.join(prep.test_data_path, 'fbank/callhome'))

    # HTK settings
    setup(corpus='switchboard', feature='fbank', dim=40, sampling_rate=8000, window=0.025, slide=0.01,
          energy=True, delta=True, deltadelta=True, window_func='hamming')

    ##############################
    # train (LDC97S62)
    ##############################
    if len(os.listdir(wav_train_path)) == 0:
        raise ValueError('Error: Convert from sph to wav files.')
    wav_train_paths = [os.path.join(wav_train_path, wav_dir)
                       for wav_dir in os.listdir(wav_train_path)]
    with open('wav2fbank_train.scp', 'w') as f:
        for wav_train_path in sorted(wav_train_paths):
            wav_index = os.path.basename(wav_train_path).split('.')[0]
            save_path = os.path.join(save_train_path, wav_index + '.htk')
            f.write(wav_train_path + '  ' + save_path + '\n')
    if len(wav_train_paths) != 4876:
        raise ValueError('Error: File number is not correct (True: 4876, Now: %d).' %
                         len(wav_train_paths))

    ##############################
    # train (Fisher)
    ##############################
    if len(os.listdir(wav_train_fisher_path)) == 0:
        raise ValueError('Error: Convert from sph to wav files.')
    wav_train_fisher_paths = [os.path.join(wav_train_fisher_path, wav_dir)
                              for wav_dir in glob.glob(os.path.join(wav_train_fisher_path, '*/*.wav'))]
    with open('wav2fbank_train_fisher.scp', 'w') as f:
        for wav_train_fisher_path in sorted(wav_train_fisher_paths):
            number = wav_train_fisher_path.split('/')[-2]
            mkdir(os.path.join(save_train_fisher_path, number))
            wav_index = os.path.basename(wav_train_fisher_path).split('.')[0]
            save_path = os.path.join(save_train_fisher_path, number, wav_index + '.htk')
            f.write(wav_train_fisher_path + '  ' + save_path + '\n')
    if len(wav_train_fisher_paths) != 23398:
        raise ValueError('Error: File number is not correct (True: 23398, Now: %d).' %
                         len(wav_train_fisher_paths))

    ##############################
    # test (eval2000, swbd)
    ##############################
    if len(os.listdir(wav_test_path)) == 0:
        raise ValueError('Error: Convert from sph to wav files.')
    wav_test_paths = [os.path.join(wav_test_path, wav_dir)
                      for wav_dir in os.listdir(wav_test_path)]
    with open('wav2fbank_test.scp', 'w') as f:
        for wav_test_path in sorted(wav_test_paths):
            wav_index = os.path.basename(wav_test_path).split('.')[0]
            save_path = os.path.join(save_test_path, wav_index + '.htk')
            f.write(wav_test_path + '  ' + save_path + '\n')
    if len(wav_test_paths) != 40:
        raise ValueError('Error: File number is not correct (True: 40, Now: %d).' %
                         len(wav_test_paths))

    ##############################
    # test (eval2000, callhome)
    ##############################
    if len(os.listdir(wav_test_callhome_path)) == 0:
        raise ValueError('Error: Convert from sph to wav files.')
    wav_test_callhome_paths = [os.path.join(wav_test_callhome_path, wav_dir)
                               for wav_dir in os.listdir(wav_test_callhome_path)]
    with open('wav2fbank_test_callhome.scp', 'w') as f:
        for wav_test_callhome_path in sorted(wav_test_callhome_paths):
            wav_index = os.path.basename(wav_test_callhome_path).split('.')[0]
            save_path = os.path.join(save_test_callhome_path, wav_index + '.htk')
            f.write(wav_test_callhome_path + '  ' + save_path + '\n')
    if len(wav_test_callhome_paths) != 40:
        raise ValueError('Error: File number is not correct (True: 40, Now: %d).' %
                         len(wav_test_callhome_paths))


if __name__ == '__main__':
    main()
