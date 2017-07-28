#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
from glob import glob

sys.path.append('../')
from inputs.input_data_global_norm import read_htk
from utils.util import mkdir_join


def main(dataset_save_path, input_feature_path):

    input_save_path = mkdir_join(dataset_save_path, 'inputs')

    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
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


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError

    dataset_save_path = args[1]
    input_feature_path = args[2]

    main(dataset_save_path, input_feature_path)
