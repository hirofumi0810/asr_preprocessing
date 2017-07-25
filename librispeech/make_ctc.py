#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for the CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isfile
import sys
from glob import glob

sys.path.append('../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.ctc.character import read_text as read_char
from labels.ctc.word import read_text as read_word
from utils.util import mkdir_join


def main(data_path, dataset_save_path, input_feature_path, run_root_path,
         label_type):

    print('===== ' + label_type + ' =====')
    prep = Prepare(data_path, run_root_path)
    input_save_path = mkdir_join(dataset_save_path, 'inputs')
    label_save_path = mkdir_join(dataset_save_path, 'labels')
    label_save_path = mkdir_join(label_save_path, 'ctc')
    label_save_path = mkdir_join(label_save_path, label_type)

    ####################
    # labels
    ####################
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        label_train_clean100_save_path = mkdir_join(
            label_save_path, 'train_clean100')
        label_train_clean360_save_path = mkdir_join(
            label_save_path, 'train_clean360')
        label_train_other500_save_path = mkdir_join(
            label_save_path, 'train_other500')
        label_dev_clean_save_path = mkdir_join(label_save_path, 'dev_clean')
        label_dev_other_save_path = mkdir_join(label_save_path, 'dev_other')
        label_test_clean_save_path = mkdir_join(label_save_path, 'test_clean')
        label_test_other_save_path = mkdir_join(label_save_path, 'test_other')

        print('=> Processing transcripts...')
        # Read target labels and save labels as npy files
        print('---------- train ----------')
        label_train_clean100_paths = prep.text(data_type='train_clean100')
        label_train_clean360_paths = prep.text(data_type='train_clean360')
        label_train_other500_paths = prep.text(data_type='train_other500')

        if label_type == 'character':
            speaker_dict_train_clean100 = read_char(
                label_paths=label_train_clean100_paths,
                run_root_path=abspath('./'),
                save_path=label_train_clean100_save_path)
            speaker_dict_train_clean360 = read_char(
                label_paths=label_train_clean360_paths,
                run_root_path=abspath('./'),
                save_path=label_train_clean360_save_path)
            speaker_dict_train_other500 = read_char(
                label_paths=label_train_other500_paths,
                run_root_path=abspath('./'),
                save_map_file=True,
                save_path=label_train_other500_save_path)
        elif label_type == 'character_capital_divide':
            speaker_dict_train_clean100 = read_char(
                label_paths=label_train_clean100_paths,
                run_root_path=abspath('./'),
                save_path=label_train_clean100_save_path,
                divide_by_capital=True)
            speaker_dict_train_clean360 = read_char(
                label_paths=label_train_clean360_paths,
                run_root_path=abspath('./'),
                save_path=label_train_clean360_save_path,
                divide_by_capital=True)
            speaker_dict_train_other500 = read_char(
                label_paths=label_train_other500_paths,
                run_root_path=abspath('./'),
                save_map_file=True,
                save_path=label_train_other500_save_path,
                divide_by_capital=True)
        elif label_type == 'word':
            speaker_dict_train_clean100 = read_word(
                label_paths=label_train_clean100_paths,
                data_type='train_clean100',
                run_root_path=abspath('./'),
                save_path=label_train_clean100_save_path,
                save_map_file=True)
            speaker_dict_train_clean360 = read_word(
                label_paths=label_train_clean360_paths,
                data_type='train_clean360',
                run_root_path=abspath('./'),
                save_path=label_train_clean360_save_path,
                save_map_file=True)
            speaker_dict_train_other500 = read_word(
                label_paths=label_train_other500_paths,
                data_type='train_other500',
                run_root_path=abspath('./'),
                save_map_file=True,
                save_path=label_train_other500_save_path)

        print('---------- dev ----------')
        label_dev_clean_paths = prep.text(data_type='dev_clean')
        label_dev_other_paths = prep.text(data_type='dev_other')
        if label_type == 'character':
            speaker_dict_dev_clean = read_char(
                label_paths=label_dev_clean_paths,
                run_root_path=abspath('./'),
                save_path=label_dev_clean_save_path)
            speaker_dict_dev_other = read_char(
                label_paths=label_dev_other_paths,
                run_root_path=abspath('./'),
                save_path=label_dev_other_save_path)
        elif label_type == 'character_capital_divide':
            speaker_dict_dev_clean = read_char(
                label_paths=label_dev_clean_paths,
                run_root_path=abspath('./'),
                save_path=label_dev_clean_save_path,
                divide_by_capital=True)
            speaker_dict_dev_other = read_char(
                label_paths=label_dev_clean_paths,
                run_root_path=abspath('./'),
                save_path=label_dev_other_save_path,
                divide_by_capital=True)
        elif label_type == 'word':
            speaker_dict_dev_clean = read_word(
                label_paths=label_dev_clean_paths,
                label_type=label_type,
                run_root_path=abspath('./'),
                save_path=label_dev_clean_save_path)
            speaker_dict_dev_other = read_word(
                label_paths=label_dev_other_paths,
                label_type=label_type,
                run_root_path=abspath('./'),
                save_path=label_dev_other_save_path)

        print('---------- test ----------')
        label_test_clean_paths = prep.text(data_type='test_clean')
        label_test_other_paths = prep.text(data_type='test_other')
        if label_type == 'character':
            speaker_dict_test_clean = read_char(
                label_paths=label_test_clean_paths,
                run_root_path=abspath('./'),
                save_path=label_test_clean_save_path)
            speaker_dict_test_other = read_char(
                label_paths=label_test_other_paths,
                run_root_path=abspath('./'),
                save_path=label_test_other_save_path)
        elif label_type == 'character_capital_divide':
            speaker_dict_test_clean = read_char(
                label_paths=label_test_clean_paths,
                run_root_path=abspath('./'),
                save_path=label_test_clean_save_path,
                divide_by_capital=True)
            speaker_dict_test_other = read_char(
                label_paths=label_test_other_paths,
                run_root_path=abspath('./'),
                save_path=label_test_other_save_path,
                divide_by_capital=True)
        elif label_type == 'word':
            speaker_dict_test_clean = read_word(
                label_paths=label_test_clean_paths,
                label_type=label_type,
                run_root_path=abspath('./'),
                save_path=label_test_clean_save_path)
            speaker_dict_test_other = read_word(
                label_paths=label_test_other_paths,
                label_type=label_type,
                run_root_path=abspath('./'),
                save_path=label_test_other_save_path)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')

    ####################
    # inputs
    ####################
    if isfile(join(input_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        input_train_clean100_save_path = mkdir_join(
            input_save_path, 'train_clean100')
        input_train_clean360_save_path = mkdir_join(
            input_save_path, 'train_clean360')
        input_train_other500_save_path = mkdir_join(
            input_save_path, 'train_other500')
        input_dev_clean_save_path = mkdir_join(input_save_path, 'dev_clean')
        input_dev_other_save_path = mkdir_join(input_save_path, 'dev_other')
        input_test_clean_save_path = mkdir_join(input_save_path, 'test_clean')
        input_test_other_save_path = mkdir_join(input_save_path, 'test_other')

        print('=> Processing input data...')
        # Read htk files, and save input data and frame num dict
        htk_train_clean100_paths = [join(input_feature_path, htk_dir)
                                    for htk_dir in sorted(glob(join(input_feature_path,
                                                                    'train_clean100/*.htk')))]
        htk_train_clean360_paths = [join(input_feature_path, htk_dir)
                                    for htk_dir in sorted(glob(join(input_feature_path,
                                                                    'train_clean360/*.htk')))]
        htk_train_other500_paths = [join(input_feature_path, htk_dir)
                                    for htk_dir in sorted(glob(join(input_feature_path,
                                                                    'train_other500/*.htk')))]
        htk_dev_clean_paths = [join(input_feature_path, htk_dir)
                               for htk_dir in sorted(glob(join(input_feature_path,
                                                               'dev_clean/*.htk')))]
        htk_dev_other_paths = [join(input_feature_path, htk_dir)
                               for htk_dir in sorted(glob(join(input_feature_path,
                                                               'dev_other/*.htk')))]
        htk_test_clean_paths = [join(input_feature_path, htk_dir)
                                for htk_dir in sorted(glob(join(input_feature_path,
                                                                'test_clean/*.htk')))]
        htk_test_other_paths = [join(input_feature_path, htk_dir)
                                for htk_dir in sorted(glob(join(input_feature_path,
                                                                'test_other/*.htk')))]

        print('---------- train ----------')
        return_tuple = read_htk(htk_paths=htk_train_clean100_paths,
                                save_path=input_train_clean100_save_path,
                                speaker_dict=speaker_dict_train_clean100,
                                normalize='speaker',
                                is_training=True)
        train_mean_male = return_tuple[0]
        train_mean_female = return_tuple[1]
        train_std_male = return_tuple[2]
        train_std_female = return_tuple[3]

        print('---------- dev ----------')
        read_htk(htk_paths=htk_dev_clean_paths,
                 save_path=input_dev_clean_save_path,
                 normalize=True,
                 is_training=False,
                 train_mean=train_mean,
                 train_std=train_std)

        print('---------- test ----------')
        read_htk(htk_paths=htk_test_clean_paths,
                 save_path=input_test_clean_save_path,
                 normalize=True,
                 is_training=False,
                 train_mean=train_mean,
                 train_std=train_std)

        # Make a confirmation file to prove that dataset was saved
        # correctly
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

    for label_type in ['character', 'character_capital_divide', 'word']:
        main(data_path,
             dataset_save_path,
             input_feature_path,
             run_root_path,
             label_type)
