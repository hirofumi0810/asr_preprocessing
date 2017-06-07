#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join, abspath
import sys
import shutil
from glob import glob
from tqdm import tqdm

sys.path.append('../')
import prepare_path
from inputs.input_data_global_norm import read_htk
from labels.ctc.character import read_text
from labels.ctc.phone import read_phone
from utils.util import mkdir_join


def main(dataset_save_path, input_feature_save_path, label_type):

    print('===== ' + label_type + ' =====')
    prep = prepare_path.Prepare(dataset_save_path, input_feature_save_path,
                                run_root_path=os.path.abspath('./'))
    save_path = mkdir_join(dataset_save_path, 'ctc')
    save_path = mkdir_join(save_path, label_type)

    # Reset directory
    if not os.path.isfile(join(save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(save_path)):
            shutil.rmtree(join(save_path, c))
    else:
        print('Already exists.')
        return 0

    train_save_path = mkdir_join(save_path, 'train')
    dev_save_path = mkdir_join(save_path, 'dev')
    test_save_path = mkdir_join(save_path, 'test')

    input_train_save_path = mkdir_join(train_save_path, 'input')
    label_train_save_path = mkdir_join(train_save_path, 'label')
    input_dev_save_path = mkdir_join(dev_save_path, 'input')
    label_dev_save_path = mkdir_join(dev_save_path, 'label')
    input_test_save_path = mkdir_join(test_save_path, 'input')
    label_test_save_path = mkdir_join(test_save_path, 'label')

    ####################
    # train
    ####################
    print('---------- train ----------')
    # Read htk files, save input data as npy files, save frame num dict as a
    # pickle file
    print('=> Processing input data...')
    htk_train_paths = [join(input_feature_save_path, htk_dir)
                       for htk_dir in sorted(glob(join(input_feature_save_path, 'train/*.htk')))]
    train_mean, train_std = read_htk(htk_paths=htk_train_paths,
                                     save_path=input_train_save_path,
                                     normalize=True,
                                     is_training=True)

    # Read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    if label_type == 'character':
        label_train_paths = prep.text(data_type='train')
        read_text(label_paths=label_train_paths,
                  run_root_path=abspath('./'),
                  save_path=label_train_save_path)
    else:
        label_train_paths = prep.phone(data_type='train')
        read_phone(label_paths=label_train_paths,
                   label_type=label_type,
                   run_root_path=abspath('./'),
                   save_path=label_train_save_path)

    ####################
    # dev
    ####################
    print('---------- dev ----------')
    # Read htk files, save input data as npy files, save frame num dict as a
    # pickle file
    print('=> Processing input data...')
    htk_dev_paths = [join(input_feature_save_path, htk_dir)
                     for htk_dir in sorted(glob(join(input_feature_save_path, 'dev/*.htk')))]
    read_htk(htk_paths=htk_dev_paths,
             save_path=input_dev_save_path,
             normalize=True,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

    # Read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    if label_type == 'character':
        label_dev_paths = prep.text(data_type='dev')
        read_text(label_paths=label_dev_paths,
                  run_root_path=abspath('./'),
                  save_path=label_dev_save_path)
    else:
        label_dev_paths = prep.phone(data_type='dev')
        read_phone(label_paths=label_dev_paths,
                   label_type=label_type,
                   run_root_path=abspath('./'),
                   save_path=label_dev_save_path)

    ####################
    # test
    ####################
    print('---------- test ----------')
    # Read htk files, save input data as npy files, save frame num dict as a
    # pickle file
    print('=> Processing input data...')
    htk_test_paths = [join(input_feature_save_path, htk_dir)
                      for htk_dir in sorted(glob(join(input_feature_save_path, 'test/*.htk')))]
    read_htk(htk_paths=htk_test_paths,
             save_path=input_test_save_path,
             normalize=True,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

    # Read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    if label_type == 'character':
        label_test_paths = prep.text(data_type='test')
        read_text(label_paths=label_test_paths,
                  run_root_path=abspath('./'),
                  save_path=label_test_save_path)
    else:
        label_test_paths = prep.phone(data_type='test')
        read_phone(label_paths=label_test_paths,
                   label_type=label_type,
                   run_root_path=abspath('./'),
                   save_path=label_test_save_path)

    # Make a confirmation file to prove that dataset was saved correctly
    with open(join(save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError

    dataset_save_path = args[1]
    input_feature_save_path = args[2]

    label_types = ['phone61', 'phone48', 'phone39', 'character']
    for label_type in label_types:
        main(dataset_save_path, input_feature_save_path, label_type)
