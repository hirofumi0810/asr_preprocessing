#! /usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join, basename
import sys
from glob import glob

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from utils.util import mkdir, mkdir_join
from utils.config.make_config import setup


def main():

    prep = Prepare()

    save_train_path = mkdir_join(prep.train_data_path, 'fbank')
    save_train_fisher_path = mkdir_join(prep.train_data_fisher_path, 'fbank')
    save_test_path = mkdir_join(prep.test_data_path, 'fbank/swbd')
    save_test_callhome_path = mkdir_join(prep.test_data_path, 'fbank/callhome')

    # HTK settings
    setup(corpus='switchboard',
          feature='fbank',
          dim=40,
          sampling_rate=8000,
          window=0.025,
          slide=0.01,
          energy=True,
          delta=True,
          deltadelta=True,
          window_func='hamming')

    ##############################
    # train (LDC97S62)
    ##############################
    wav_paths = [join(prep.train_data_path, wav_dir)
                 for wav_dir in glob(join(prep.train_data_path,
                                          'wav/*.wav'))]
    if len(wav_paths) == 0:
        raise ValueError('Convert from sph to wav files.')
    elif len(wav_paths) != 4876:
        raise ValueError(("File number is not correct "
                          "(True: 4876, Now: %d).") % len(wav_paths))
    with open('wav2fbank_train.scp', 'w') as f:
        for wav_path in sorted(wav_paths):
            wav_index = basename(wav_path).split('.')[0]
            save_path = join(save_train_path, wav_index + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ##############################
    # train (Fisher)
    ##############################
    wav_paths = [join(prep.train_data_fisher_path, wav_dir)
                 for wav_dir in glob(join(prep.train_data_fisher_path,
                                          'wav/*/*.wav'))]
    if len(wav_paths) == 0:
        raise ValueError('Convert from sph to wav files.')
    elif len(wav_paths) != 23398:
        raise ValueError(("File number is not correct "
                          "(True: 23398, Now: %d).") % len(wav_paths))
    with open('wav2fbank_train_fisher.scp', 'w') as f:
        for wav_path in sorted(wav_paths):
            number = wav_path.split('/')[-2]
            mkdir(save_train_fisher_path, number)
            wav_index = basename(wav_path).split('.')[0]
            save_path = join(save_train_fisher_path,
                             number, wav_index + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ##############################
    # test (eval2000, swbd)
    ##############################
    wav_paths = [join(prep.test_data_path, wav_dir)
                 for wav_dir in glob(join(prep.test_data_path, 'wav/swbd/*.wav'))]
    if len(wav_paths) == 0:
        raise ValueError('Convert from sph to wav files.')
    elif len(wav_paths) != 40:
        raise ValueError(("File number is not correct"
                          "(True: 40, Now: %d).") % len(wav_paths))
    with open('wav2fbank_test.scp', 'w') as f:
        for wav_path in sorted(wav_paths):
            wav_index = basename(wav_path).split('.')[0]
            save_path = join(save_test_path, wav_index + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')

    ##############################
    # test (eval2000, callhome)
    ##############################
    wav_paths = [join(prep.test_data_path, wav_dir)
                 for wav_dir in glob(join(prep.test_data_path,
                                          'wav/callhome/*.wav'))]
    if len(wav_paths) == 0:
        raise ValueError('Convert from sph to wav files.')
    elif len(wav_paths) != 40:
        raise ValueError(("File number is not correct ",
                          "(True: 40, Now: %d).") % len(wav_paths))
    with open('wav2fbank_test_callhome.scp', 'w') as f:
        for wav_path in sorted(wav_paths):
            wav_index = basename(wav_path).split('.')[0]
            save_path = join(save_test_callhome_path, wav_index + '.htk')
            f.write(wav_path + '  ' + save_path + '\n')


if __name__ == '__main__':
    main()
