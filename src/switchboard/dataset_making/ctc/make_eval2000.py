#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC network (eval2000-swbd corpus)."""

import os
import sys
import shutil
from glob import glob
import numpy as np
from tqdm import tqdm

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.ctc.eval2000.swbd import read_txt
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
                os.remove(os.path.join(test_save_path, c))
    else:
        print('Already exists.')
        return 0

    input_save_path = mkdir(os.path.join(test_save_path, 'input'))
    label_save_path = mkdir(os.path.join(test_save_path, 'label'))

    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_paths = prep.label_test()
    speaker_dict = read_txt(label_paths=label_paths,
                            save_path=label_save_path)

    # load statistics over train dataset
    train_mean = np.load(os.path.join(prep.dataset_path,
                                      'ctc/character/train/train_mean.npy'))
    train_std = np.load(os.path.join(prep.dataset_path,
                                     'ctc/character/train/train_std.npy'))

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.test_data_path, htk_dir)
                 for htk_dir in sorted(glob(os.path.join(prep.test_data_path,
                                                         'fbank/swbd/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_save_path,
             speaker_dict=speaker_dict,
             global_norm=False,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

    # make a confirmation file to prove that dataset was saved correctly
    with open(os.path.join(test_save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    print('===========================================================')
    print('=                     eval2000 (swbd)                     =')
    print('===========================================================')

    main()
