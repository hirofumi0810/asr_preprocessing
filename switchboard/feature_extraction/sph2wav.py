#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert audio files from sph to wav."""

import os
from os.path import join
import sys
from glob import glob
import shutil
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from utils.util import mkdir, mkdir_join


def main():

    prep = Prepare()
    sph_train_paths = prep.sph_train('ldc97s62')
    sph_train_fisher_paths = prep.sph_train('fisher')
    sph_test_paths = prep.sph_test('swbd')
    sph_test_callhome_paths = prep.sph_test('callhome')

    wav_train_path = mkdir_join(prep.train_data_path, 'wav')
    wav_train_fisher_path = mkdir_join(prep.train_data_fisher_path, 'wav')
    wav_test_path = mkdir_join(prep.test_data_path, 'wav')
    wav_test_swbd_path = mkdir_join(wav_test_path, 'swbd')
    wav_test_callhome_path = mkdir_join(wav_test_path, 'callhome')

    # train (ldc97s62)
    print('===== ldc97s62 =====')
    if len(os.listdir(wav_train_path)) == 4876:
        print('Already converted.')
    else:
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(wav_train_path)):
            os.remove(join(wav_train_path, c))

        for sph_train_path in tqdm(sph_train_paths):
            wav_index = os.path.basename(sph_train_path).split('.')[0]
            save_path_a = join(wav_train_path, wav_index + 'A.wav')
            save_path_b = join(wav_train_path, wav_index + 'B.wav')

            # convert from sph to wav
            # A side
            os.system('~/tool/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 %s %s' %
                      (sph_train_path, save_path_a))
            # B side
            os.system('~/tool/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 %s %s' %
                      (sph_train_path, save_path_b))

    # train (Fisher)
    print('===== Fisher =====')
    if len(glob(join(wav_train_fisher_path, '*/*.wav'))) == 23398:
        print('Already converted.')
    else:
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(wav_train_fisher_path)):
            shutil.rmtree(join(wav_train_fisher_path, c))

        for sph_train_fisher_path in tqdm(sph_train_fisher_paths):
            number = sph_train_fisher_path.split('/')[-2]
            mkdir(wav_train_fisher_path, number)
            wav_index = os.path.basename(sph_train_fisher_path).split('.')[0]
            save_path_a = join(wav_train_fisher_path,
                               number, wav_index + 'A.wav')
            save_path_b = join(
                wav_train_fisher_path, number, wav_index + 'B.wav')

            # convert from sph to wav
            # A side
            os.system('~/tool/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 %s %s' %
                      (sph_train_fisher_path, save_path_a))
            # B side
            os.system('~/tool/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 %s %s' %
                      (sph_train_fisher_path, save_path_b))

    # test (eval2000, swbd)
    print('===== eval2000 (swbd) =====')
    if len(os.listdir(wav_test_swbd_path)) == 40:
        print('Already converted.')
    else:
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(wav_test_swbd_path)):
            os.remove(join(wav_test_swbd_path, c))

        for sph_test_path in tqdm(sph_test_paths):
            wav_index = os.path.basename(sph_test_path).split('.')[0]
            save_path_a = join(wav_test_swbd_path, wav_index + 'A.wav')
            save_path_b = join(wav_test_swbd_path, wav_index + 'B.wav')

            # convert from sph to wav
            # A side
            os.system('~/tool/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 %s %s' %
                      (sph_test_path, save_path_a))
            # B side
            os.system('~/tool/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 %s %s' %
                      (sph_test_path, save_path_b))

    # test (eval2000, callhome)
    print('===== eval2000 (callhome) =====')
    if len(os.listdir(wav_test_callhome_path)) == 40:
        print('Already converted.')
    else:
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(wav_test_callhome_path)):
            os.remove(join(wav_test_callhome_path, c))

        for sph_test_callhome_path in tqdm(sph_test_callhome_paths):
            wav_index = os.path.basename(sph_test_callhome_path).split('.')[0]
            save_path_a = join(wav_test_callhome_path, wav_index + 'A.wav')
            save_path_b = join(wav_test_callhome_path, wav_index + 'B.wav')

            # convert from sph to wav
            # A side
            os.system('~/tool/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 %s %s' %
                      (sph_test_callhome_path, save_path_a))
            # B side
            os.system('~/tool/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 %s %s' %
                      (sph_test_callhome_path, save_path_b))


if __name__ == '__main__':
    main()
