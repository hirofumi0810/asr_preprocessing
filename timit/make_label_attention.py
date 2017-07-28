#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make labels for the Attention model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isfile
import sys

sys.path.append('../')
from prepare_path import Prepare
from labels.attention.character import read_text
from labels.attention.phone import read_phone
from utils.util import mkdir_join


def main(data_path, dataset_save_path, run_root_path, label_type):

    print('===== ' + label_type + ' =====')
    prep = Prepare(data_path, run_root_path)
    label_save_path = mkdir_join(dataset_save_path, 'labels')
    label_save_path = mkdir_join(label_save_path, 'attention')
    label_save_path = mkdir_join(label_save_path, label_type)

    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.')
    else:
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
        elif label_type == 'character_capital_divide':
            label_train_paths = prep.text(data_type='train')
            read_text(label_paths=label_train_paths,
                      run_root_path=abspath('./'),
                      save_map_file=True,
                      save_path=label_train_save_path,
                      divide_by_capital=True)
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
        elif label_type == 'character_capital_divide':
            label_dev_paths = prep.text(data_type='dev')
            read_text(label_paths=label_dev_paths,
                      run_root_path=abspath('./'),
                      save_path=label_dev_save_path,
                      divide_by_capital=True)
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
        elif label_type == 'character_capital_divide':
            label_test_paths = prep.text(data_type='test')
            read_text(label_paths=label_test_paths,
                      run_root_path=abspath('./'),
                      save_path=label_test_save_path,
                      divide_by_capital=True)
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
    if len(args) != 4:
        raise ValueError

    data_path = args[1]
    dataset_save_path = args[2]
    run_root_path = args[3]

    for label_type in ['character', 'character_capital_divide', 'phone61', 'phone48', 'phone39']:
        main(data_path,
             dataset_save_path,
             run_root_path,
             label_type)
