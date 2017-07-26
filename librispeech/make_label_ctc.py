#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make labels for the CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys

sys.path.append('../')
from prepare_path import Prepare
from labels.ctc.character import read_text as read_char
from labels.ctc.word import read_text as read_word
from utils.util import mkdir_join


def main(data_path, dataset_save_path, run_root_path, label_type,
         train_data_type):

    print('===== train_data_type: %s, label_type: %s =====' %
          (train_data_type, label_type))

    prep = Prepare(data_path, run_root_path)
    label_save_path = mkdir_join(dataset_save_path, 'labels')
    label_save_path = mkdir_join(label_save_path, 'ctc')
    label_save_path = mkdir_join(label_save_path, train_data_type)
    label_save_path = mkdir_join(label_save_path, label_type)

    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        label_train_save_path = mkdir_join(label_save_path, train_data_type)
        label_dev_clean_save_path = mkdir_join(label_save_path, 'dev_clean')
        label_dev_other_save_path = mkdir_join(label_save_path, 'dev_other')
        label_test_clean_save_path = mkdir_join(label_save_path, 'test_clean')
        label_test_other_save_path = mkdir_join(label_save_path, 'test_other')

        if label_type == 'character_capital_divide':
            divide_by_capital = True
        else:
            divide_by_capital = False

        print('=> Processing transcripts...')
        # Read target labels and save labels as npy files
        print('---------- train_ ----------')
        label_train_paths = prep.text(data_type=train_data_type)
        if label_type == 'word':
            read_word(label_paths=label_train_paths,
                      data_type=train_data_type,
                      train_data_type=train_data_type,
                      run_root_path=prep.run_root_path,
                      save_map_file=True,
                      save_path=label_train_save_path,
                      frequency_threshold=10)
        else:
            read_char(label_paths=label_train_paths,
                      run_root_path=prep.run_root_path,
                      save_path=label_train_save_path,
                      divide_by_capital=divide_by_capital)

        print('---------- dev_clean ----------')
        label_dev_clean_paths = prep.text(data_type='dev_clean')
        if label_type == 'word':
            read_word(label_paths=label_dev_clean_paths,
                      data_type='dev_clean',
                      train_data_type=train_data_type,
                      run_root_path=prep.run_root_path,
                      save_path=label_dev_clean_save_path)
        else:
            read_char(label_paths=label_dev_clean_paths,
                      run_root_path=prep.run_root_path,
                      save_path=label_dev_clean_save_path,
                      divide_by_capital=divide_by_capital)

        print('---------- dev_other ----------')
        label_dev_other_paths = prep.text(data_type='dev_other')
        if label_type == 'word':
            read_word(label_paths=label_dev_other_paths,
                      data_type='dev_other',
                      train_data_type=train_data_type,
                      run_root_path=prep.run_root_path,
                      save_path=label_dev_other_save_path)
        else:
            read_char(label_paths=label_dev_clean_paths,
                      run_root_path=prep.run_root_path,
                      save_path=label_dev_other_save_path,
                      divide_by_capital=divide_by_capital)

        print('---------- test_clean ----------')
        label_test_clean_paths = prep.text(data_type='test_clean')
        if label_type == 'word':
            read_word(label_paths=label_test_clean_paths,
                      data_type='test_clean',
                      train_data_type=train_data_type,
                      run_root_path=prep.run_root_path,
                      save_path=label_test_clean_save_path)
        else:
            read_char(label_paths=label_test_clean_paths,
                      run_root_path=prep.run_root_path,
                      save_path=label_test_clean_save_path,
                      divide_by_capital=divide_by_capital)

        print('---------- test_other ----------')
        label_test_other_paths = prep.text(data_type='test_other')
        if label_type == 'word':
            read_word(label_paths=label_test_other_paths,
                      data_type='test_other',
                      train_data_type=train_data_type,
                      run_root_path=prep.run_root_path,
                      save_path=label_test_other_save_path)
        else:
            read_char(label_paths=label_test_clean_paths,
                      run_root_path=prep.run_root_path,
                      save_path=label_test_other_save_path,
                      divide_by_capital=divide_by_capital)

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

    for train_data_type in ['train_clean100', 'train_clean360', 'train_other500']:
        for label_type in ['character', 'character_capital_divide', 'word']:
            main(data_path,
                 dataset_save_path,
                 run_root_path,
                 label_type,
                 train_data_type)
