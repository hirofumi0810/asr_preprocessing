#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isfile
import sys
import argparse

sys.path.append('../')
from prepare_path import Prepare
from labels.character import read_text
from labels.phone import read_phone
from utils.util import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to TIMIT dataset')
parser.add_argument('--dataset_save_path', type=str, help='path to save dataset')
parser.add_argument('--run_root_path', type=str, help='path to run this script')


def main(model, label_type):

    print('==================================================')
    print('  model: %s' % model)
    print('  label_type: %s' % label_type)
    print('==================================================')

    args = parser.parse_args()
    prep = Prepare(args.data_path, args.run_root_path)
    label_save_path = mkdir_join(args.dataset_save_path, 'labels', model, label_type)

    print('=> Processing transcripts...')
    if isfile(join(label_save_path, 'complete.txt')):
        print('Already exists.\n')
    else:
        for data_type in ['train', 'dev', 'test']:
            save_map_file = True if data_type == 'train' else False

            # Read target labels and save labels as npy files
            print('---------- %s ----------' % data_type)
            if label_type in ['character', 'character_capital_divide']:
                divide_by_capital = False if label_type == 'character' else True
                read_text(label_paths=prep.text(data_type=data_type),
                          run_root_path=abspath('./'),
                          model=model,
                          save_map_file=save_map_file,
                          save_path=mkdir_join(label_save_path, data_type),
                          divide_by_capital=divide_by_capital)
            else:
                # 39 or 48 or 61 phones
                read_phone(label_paths=prep.phone(data_type=data_type),
                           label_type=label_type,
                           run_root_path=abspath('./'),
                           model=model,
                           save_map_file=save_map_file,
                           save_path=mkdir_join(label_save_path, data_type))

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')


if __name__ == '__main__':
    for model in ['ctc', 'attention']:
        for label_type in ['character', 'character_capital_divide',
                           'phone61', 'phone48', 'phone39']:
            main(model, label_type)
