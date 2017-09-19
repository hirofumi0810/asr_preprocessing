#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make labels for the CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import argparse

sys.path.append('../')
from prepare_path import Prepare
from labels.ctc.character import read_text as read_char
from labels.ctc.word import read_text as read_word
from utils.util import mkdir_join


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to Librispeech dataset')
parser.add_argument('--dataset_save_path', type=str, help='path to save dataset')
parser.add_argument('--run_root_path', type=str, help='path to run this script')


def main(label_type, train_data_size, label_type):

    print('==================================================')
    print('  model: %s' % model)
    print('  train_data_size: %s' % train_data_size)
    print('  label_type: %s' % label_type)
    print('==================================================')

    args = parser.parse_args()
    prep = Prepare(args.data_path, args.run_root_path)
    label_save_path = mkdir_join(args.dataset_save_path, 'labels',
                                 model, train_data_size, label_type)

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.')
    else:
        divide_by_capital = True label_type == 'character_capital_divide' else False

        # Read target labels and save labels as npy files
        print('---------- train_ ----------')
        label_paths = prep.text(data_type=train_data_size)
        if label_type == 'word':
            read_word(label_paths=label_paths,
                      data_type=train_data_size,
                      train_data_size=train_data_size,
                      run_root_path=prep.run_root_path,
                      save_map_file=True,
                      save_path=mkdir_join(label_save_path, train_data_size),
                      frequency_threshold=10)
        else:
            read_char(label_paths=label_paths,
                      run_root_path=prep.run_root_path,
                      save_path=mkdir_join(label_save_path, train_data_size),
                      divide_by_capital=divide_by_capital)

        for data_type in ['dev_clean', 'dev_other', 'test_clean', 'test_other']:

            # Read target labels and save labels as npy files
            print('---------- %s ----------' % data_type)
            if label_type == 'word':
                read_word(label_paths=prep.text(data_type=data_type),
                          data_type=data_type,
                          train_data_size=train_data_size,
                          run_root_path=prep.run_root_path,
                          save_path=mkdir_join(label_save_path, data_type))
            else:
                read_char(label_paths=prep.text(data_type=data_type),
                          run_root_path=prep.run_root_path,
                          save_path=mkdir_join(label_save_path, data_type),
                          divide_by_capital=divide_by_capital)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':

    for model in ['ctc', 'attention']:
        for train_data_size in ['train_clean100', 'train_clean360',
                                'train_other500', 'train_all']:
            for label_type in ['character', 'character_capital_divide', 'word']:
                main(model, train_data_size, label_type)
