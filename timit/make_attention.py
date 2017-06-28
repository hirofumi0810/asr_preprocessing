#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for the Attention model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isfile
import sys
import shutil
from glob import glob


sys.path.append('../')
import prepare_path
from inputs.input_data_global_norm import read_htk
from labels.attention.character import read_text
from labels.attention.phone import read_phone
from utils.util import mkdir_join


def main(timit_path, dataset_save_path, input_feature_path, run_root_path,
         label_type):

    print('===== ' + label_type + ' =====')
    prep = prepare_path.Prepare(timit_path, run_root_path)
    input_save_path = mkdir_join(dataset_save_path, 'inputs')
    label_save_path = mkdir_join(dataset_save_path, 'labels')
    label_save_path = mkdir_join(label_save_path, 'attention')
    label_save_path = mkdir_join(label_save_path, label_type)

    ####################
    # inputs
    ####################
    if not isfile(join(input_save_path, 'complete.txt')):
        print('=> Deleting old dataset...')
        shutil.rmtree(input_save_path)

        input_save_path = mkdir_join(dataset_save_path, 'inputs')
        input_train_save_path = mkdir_join(input_save_path, 'train')
        input_dev_save_path = mkdir_join(input_save_path, 'dev')
        input_test_save_path = mkdir_join(input_save_path, 'test')

        print('=> Processing input data...')
        # Read htk files, and save input data and frame num dict
        htk_train_paths = [join(input_feature_path, htk_dir)
                           for htk_dir in sorted(glob(join(input_feature_path,
                                                           'train/*.htk')))]
        htk_dev_paths = [join(input_feature_path, htk_dir)
                         for htk_dir in sorted(glob(join(input_feature_path,
                                                         'dev/*.htk')))]
        htk_test_paths = [join(input_feature_path, htk_dir)
                          for htk_dir in sorted(glob(join(input_feature_path,
                                                          'test/*.htk')))]

        print('---------- train ----------')
        train_mean, train_std = read_htk(htk_paths=htk_train_paths,
                                         save_path=input_train_save_path,
                                         normalize=True,
                                         is_training=True)

        print('---------- dev ----------')
        read_htk(htk_paths=htk_dev_paths,
                 save_path=input_dev_save_path,
                 normalize=True,
                 is_training=False,
                 train_mean=train_mean,
                 train_std=train_std)

        print('---------- test ----------')
        read_htk(htk_paths=htk_test_paths,
                 save_path=input_test_save_path,
                 normalize=True,
                 is_training=False,
                 train_mean=train_mean,
                 train_std=train_std)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'complete.txt'), 'w') as f:
            f.write('')

    ####################
    # labels
    ####################
    if not isfile(join(label_save_path, 'complete.txt')):
        print('=> Deleting old dataset...')
        shutil.rmtree(label_save_path)

        label_save_path = mkdir_join(dataset_save_path, 'labels')
        label_save_path = mkdir_join(label_save_path, 'attention')
        label_save_path = mkdir_join(label_save_path, label_type)

        label_train_save_path = mkdir_join(label_save_path, 'train')
        label_dev_save_path = mkdir_join(label_save_path, 'dev')
        label_test_save_path = mkdir_join(label_save_path, 'test')

        print('=> Processing transcripts...')
        # Read target labels and save labels as npy files
        print('---------- train ----------')
        if label_type == 'character':
            label_train_paths = prep.text(data_type='train')
            read_text(label_paths=label_train_paths,
                      run_root_path=abspath('./'),
                      save_map_file=True,
                      save_path=label_train_save_path)
        else:
            label_train_paths = prep.phone(data_type='train')
            read_phone(label_paths=label_train_paths,
                       label_type=label_type,
                       run_root_path=abspath('./'),
                       save_map_file=True,
                       save_path=label_train_save_path)

        print('---------- dev ----------')
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

        print('---------- test ----------')
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
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 5:
        raise ValueError

    timit_path = args[1]
    dataset_save_path = args[2]
    input_feature_path = args[3]
    run_root_path = args[4]

    for label_type in ['character', 'phone61', 'phone48', 'phone39']:
        main(timit_path,
             dataset_save_path,
             input_feature_path,
             run_root_path,
             label_type)
