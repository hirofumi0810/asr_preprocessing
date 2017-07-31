#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input data (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
from glob import glob

sys.path.append('../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from utils.util import mkdir_join


def main(data_path, dataset_save_path, input_feature_path, run_root_path,
         train_data_size):

    print('===== ' + train_data_size + ' =====')
    prep = Prepare(data_path, run_root_path)
    input_save_path = mkdir_join(dataset_save_path, 'inputs')
    input_save_path = mkdir_join(input_save_path, train_data_size)

    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        input_train_save_path = mkdir_join(input_save_path, train_data_size)
        input_dev_clean_save_path = mkdir_join(input_save_path, 'dev_clean')
        input_dev_other_save_path = mkdir_join(input_save_path, 'dev_other')
        input_test_clean_save_path = mkdir_join(input_save_path, 'test_clean')
        input_test_other_save_path = mkdir_join(input_save_path, 'test_other')

        print('=> Processing input data...')
        # Read htk files, and save input data and frame num dict
        if train_data_size == 'train_all':
            htk_train_paths = [
                join(input_feature_path, htk_path)
                for htk_path in sorted(glob(join(input_feature_path,
                                                 'train_clean100/*/*.htk')))]
            htk_train_paths += [
                join(input_feature_path, htk_path)
                for htk_path in sorted(glob(join(input_feature_path,
                                                 'train_clean360/*/*.htk')))]
            htk_train_paths += [
                join(input_feature_path, htk_path)
                for htk_path in sorted(glob(join(input_feature_path,
                                                 'train_other500/*/*.htk')))]
        else:
            htk_train_paths = [
                join(input_feature_path, htk_dir)
                for htk_dir in sorted(glob(join(input_feature_path,
                                                train_data_size + '/*/*.htk')))]
        htk_dev_clean_paths = [
            join(input_feature_path, htk_dir)
            for htk_dir in sorted(glob(join(input_feature_path,
                                            'dev_clean/*/*.htk')))]
        htk_dev_other_paths = [
            join(input_feature_path, htk_dir)
            for htk_dir in sorted(glob(join(input_feature_path,
                                            'dev_other/*/*.htk')))]
        htk_test_clean_paths = [
            join(input_feature_path, htk_dir)
            for htk_dir in sorted(glob(join(input_feature_path,
                                            'test_clean/*/*.htk')))]
        htk_test_other_paths = [
            join(input_feature_path, htk_dir)
            for htk_dir in sorted(glob(join(input_feature_path,
                                            'test_other/*/*.htk')))]

        print('---------- train ----------')
        train_mean_male, train_mean_female, train_std_male, train_std_female = read_htk(
            htk_paths=htk_train_paths,
            normalize='speaker',
            is_training=True,
            speaker_gender_dict=prep.speaker_gender_dict,
            save_path=input_train_save_path)

        print('---------- dev_clean ----------')
        read_htk(htk_paths=htk_dev_clean_paths,
                 normalize='speaker',
                 is_training=False,
                 speaker_gender_dict=prep.speaker_gender_dict,
                 save_path=input_dev_clean_save_path,
                 train_mean_male=train_mean_male,
                 train_mean_female=train_mean_female,
                 train_std_male=train_std_female,
                 train_std_female=train_std_female)

        print('---------- dev_other ----------')
        read_htk(htk_paths=htk_dev_other_paths,
                 normalize='speaker',
                 is_training=False,
                 speaker_gender_dict=prep.speaker_gender_dict,
                 save_path=input_dev_other_save_path,
                 train_mean_male=train_mean_male,
                 train_mean_female=train_mean_female,
                 train_std_male=train_std_female,
                 train_std_female=train_std_female)

        print('---------- test_clean ----------')
        read_htk(htk_paths=htk_test_clean_paths,
                 normalize='speaker',
                 is_training=False,
                 speaker_gender_dict=prep.speaker_gender_dict,
                 save_path=input_test_clean_save_path,
                 train_mean_male=train_mean_male,
                 train_mean_female=train_mean_female,
                 train_std_male=train_std_female,
                 train_std_female=train_std_female)

        print('---------- test_other ----------')
        read_htk(htk_paths=htk_test_other_paths,
                 normalize='speaker',
                 is_training=False,
                 speaker_gender_dict=prep.speaker_gender_dict,
                 save_path=input_test_other_save_path,
                 train_mean_male=train_mean_male,
                 train_mean_female=train_mean_female,
                 train_std_male=train_std_female,
                 train_std_female=train_std_female)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 5:
        raise ValueError

    data_path = args[1]
    dataset_save_path = args[2]
    input_feature_path = args[3]
    run_root_path = args[4]

    for train_data_size in ['train_clean100', 'train_clean360',
                            'train_other500', 'train_all']:
        main(data_path,
             dataset_save_path,
             input_feature_path,
             run_root_path,
             train_data_size)
