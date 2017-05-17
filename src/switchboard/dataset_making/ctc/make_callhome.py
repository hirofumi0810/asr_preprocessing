#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC network (callhome corpus)."""

import os
import sys
import shutil
import glob
import numpy as np
from tqdm import tqdm

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data_global_norm import read_htk
from labels.ctc.eval2000.callhome import read_transcript
from utils.util import mkdir


def main():

    print('========== swbd ==========')
    prep = Prepare()
    save_path = mkdir(os.path.join(prep.dataset_path, 'ctc'))
    test_save_path = mkdir(os.path.join(save_path, 'test'))
    test_save_path = mkdir(os.path.join(test_save_path, 'swbd'))

    # reset directory
    if not os.path.isfile(os.path.join(test_save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(test_save_path)):
            try:
                shutil.rmtree(os.path.join(test_save_path, c))
            except:
                os.remove(c)
    else:
        print('Already exists.')
        return 0

    input_save_path = mkdir(os.path.join(test_save_path, 'input'))
    label_save_path = mkdir(os.path.join(test_save_path, 'label'))

    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_paths = prep.label_test()
    speaker_dict = read_transcript(label_paths=label_paths,
                                   save_path=label_save_path)

    # load statistics over train dataset
    train_mean = np.load(os.path.join(save_path, 'train/train_mean.npy'))
    train_std = np.load(os.path.join(save_path, 'train/train_std.npy'))

    # read htk files, save input data as npy files, save frame num dict as a
    # pickle file
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.test_data_path, htk_dir)
                 for htk_dir in sorted(glob.glob(os.path.join(prep.test_data_path, 'fbank/swbd/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_save_path,
             speaker_dict=speaker_dict,
             normalize=True,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

    # make a confirmation file to prove that dataset was saved correctly
    with open(os.path.join(test_save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


def main_callhome():

    print('========== callhome ==========')
    prep = Prepare()
    save_path = mkdir(os.path.join(prep.root_path, 'dataset'))
    test_save_path = mkdir(os.path.join(save_path, 'test'))
    test_save_path = mkdir(os.path.join(save_path, 'callhome'))

    # reset directory
    if not os.path.isfile(os.path.join(test_save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in os.listdir(test_save_path):
            shutil.rmtree(os.path.join(test_save_path, c))
    else:
        print('Already exists.')
        return 0

    input_save_path = mkdir(os.path.join(test_save_path, 'input'))
    label_save_path = mkdir(os.path.join(test_save_path, 'label'))

    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_paths = prep.label_test_callhome()
    speaker_dict = read_transcript(label_paths=label_paths,
                                   save_path=label_save_path)

    # load statistics over train dataset
    train_mean = np.load(os.path.join(prep.dataset_path,
                                      'ctc/character/train/train_mean.npy'))
    train_std = np.load(os.path.join(prep.dataset_path,
                                     'ctc/character/train/train_std.npy'))

    # read htk files, save input data as npy files, save frame num dict as a
    # pickle file
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.test_data_path, htk_dir)
                 for htk_dir in sorted(glob.glob(os.path.join(prep.test_data_path, 'fbank/callhome/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_save_path,
             speaker_dict=speaker_dict,
             normalize=True,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

    # make a confirmation file to prove that dataset was saved correctly
    with open(os.path.join(test_save_path, 'check.txt'), 'w') as f:
        f.write('')


if __name__ == '__main__':

    print('===========================================================')
    print('=                     eval2000 (swbd)                     =')
    print('===========================================================')

    main()
