#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC model (CSJ corpus)."""

import os
import sys
import shutil
import glob
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
# from inputs.input_data_global_norm import read_htk
from labels.ctc.monolog.character import read_sdb
from utils.util import mkdir


def main(label_type):

    print('===== ' + label_type + ' =====')
    prep = Prepare()
    save_path = mkdir(os.path.join(prep.dataset_path, 'ctc'))
    save_path = mkdir(os.path.join(save_path, label_type))

    # reset directory
    if not os.path.isfile(os.path.join(save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(save_path)):
            shutil.rmtree(os.path.join(save_path, c))
    else:
        print('Already exists.')
        return 0

    train_save_path = mkdir(os.path.join(save_path, 'train'))
    train_plus_save_path = mkdir(os.path.join(save_path, 'train_plus'))
    dev_save_path = mkdir(os.path.join(save_path, 'dev'))
    eval1_save_path = mkdir(os.path.join(save_path, 'eval1'))
    eval2_save_path = mkdir(os.path.join(save_path, 'eval2'))
    eval3_save_path = mkdir(os.path.join(save_path, 'eval3'))

    input_train_save_path = mkdir(os.path.join(train_save_path, 'input'))
    label_train_save_path = mkdir(os.path.join(train_save_path, 'label'))
    input_train_plus_save_path = mkdir(
        os.path.join(train_plus_save_path, 'input'))
    label_train_plus_save_path = mkdir(
        os.path.join(train_plus_save_path, 'label'))
    input_dev_save_path = mkdir(os.path.join(dev_save_path, 'input'))
    label_dev_save_path = mkdir(os.path.join(dev_save_path, 'label'))
    input_eval1_save_path = mkdir(os.path.join(eval1_save_path, 'input'))
    label_eval1_save_path = mkdir(os.path.join(eval1_save_path, 'label'))
    input_eval2_save_path = mkdir(os.path.join(eval2_save_path, 'input'))
    label_eval2_save_path = mkdir(os.path.join(eval2_save_path, 'label'))
    input_eval3_save_path = mkdir(os.path.join(eval3_save_path, 'input'))
    label_eval3_save_path = mkdir(os.path.join(eval3_save_path, 'label'))

    #####################
    # train (defalt set)
    #####################
    print('---------- train (defalt set) ----------')
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_train_paths = prep.trans(data_type='train')
    speaker_dict = read_sdb(label_paths=label_train_paths,
                            label_type=label_type,
                            save_path=label_train_save_path)

    ####################
    # train (plus)
    ####################
    print('---------- train (plus) ----------')
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_train_plus_paths = prep.trans(data_type='train_plus')
    speaker_dict = read_sdb(label_paths=label_train_plus_paths,
                            label_type=label_type,
                            save_path=label_train_plus_save_path)

    ####################
    # dev
    ####################
    print('---------- dev ----------')
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_dev_paths = prep.trans(data_type='dev')
    speaker_dict = read_sdb(label_paths=label_dev_paths,
                            label_type=label_type,
                            save_path=label_dev_save_path)

    ####################
    # eval1
    ####################
    print('---------- eval1 ----------')
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval1_paths = prep.trans(data_type='eval1')
    speaker_dict = read_sdb(label_paths=label_eval1_paths,
                            label_type=label_type,
                            save_path=label_eval1_save_path)

    ####################
    # eval2
    ####################
    print('---------- eval2 ----------')
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval2_paths = prep.trans(data_type='eval2')
    speaker_dict = read_sdb(label_paths=label_eval2_paths,
                            label_type=label_type,
                            save_path=label_eval2_save_path)

    ####################
    # eval3
    ####################
    print('---------- eval3 ----------')
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval3_paths = prep.trans(data_type='eval3')
    speaker_dict = read_sdb(label_paths=label_eval3_paths,
                            label_type=label_type,
                            save_path=label_eval3_save_path)

    # make a confirmation file to prove that dataset was saved correctly
    # with open(os.path.join(save_path, 'check.txt'), 'w') as f:
    #     f.write('')
    # print('Successfully completed!')


if __name__ == '__main__':

    print('=================================')
    print('=           CSJ (CTC)           =')
    print('=================================')

    # label_types = ['character', 'phone']
    label_types = ['phone']

    for label_type in label_types:
        main(label_type)
