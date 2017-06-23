#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for Attention model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
import sys
import shutil
from glob import glob
from tqdm import tqdm


sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.attention.character import read_sdb
from utils.util import mkdir_join


def main(csj_path, dataset_save_path, input_feature_path, run_root_path,
         label_type, train_data_type):

    print('===== ' + label_type + ' (' + train_data_type + ') =====')
    prep = Prepare(csj_path, run_root_path)
    save_path = mkdir_join(dataset_save_path, 'attention')
    save_path = mkdir_join(save_path, label_type)
    save_path = mkdir_join(save_path, train_data_type)

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
    eval1_save_path = mkdir_join(save_path, 'eval1')
    eval2_save_path = mkdir_join(save_path, 'eval2')
    eval3_save_path = mkdir_join(save_path, 'eval3')

    input_train_save_path = mkdir_join(train_save_path, 'input')
    label_train_save_path = mkdir_join(train_save_path, 'label')
    input_dev_save_path = mkdir_join(dev_save_path, 'input')
    label_dev_save_path = mkdir_join(dev_save_path, 'label')
    input_eval1_save_path = mkdir_join(eval1_save_path, 'input')
    label_eval1_save_path = mkdir_join(eval1_save_path, 'label')
    input_eval2_save_path = mkdir_join(eval2_save_path, 'input')
    label_eval2_save_path = mkdir_join(eval2_save_path, 'label')
    input_eval3_save_path = mkdir_join(eval3_save_path, 'input')
    label_eval3_save_path = mkdir_join(eval3_save_path, 'label')

    ################
    # train
    ################
    print('---------- train ----------')
    # Load target labels and save labels as npy files
    print('=> Processing transcripts...')
    if train_data_type == 'default':
        data_type = 'train'
        save_map_file = False
    elif train_data_type == 'large':
        data_type = 'train_large'
        save_map_file = True
    label_train_paths = prep.trans(data_type=data_type)
    speaker_dict = read_sdb(label_paths=label_train_paths,
                            label_type=label_type,
                            run_root_path=run_root_path,
                            save_path=label_train_save_path,
                            save_map_file=save_map_file)

    # Load htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(input_feature_path, htk_dir)
                 for htk_dir in sorted(glob(join(input_feature_path,
                                                 data_type, '*.htk')))]
    return_tuple = read_htk(htk_paths=htk_paths,
                            save_path=input_train_save_path,
                            speaker_dict=speaker_dict,
                            normalize='speaker',
                            is_training=True)
    train_mean_male, train_mean_female, train_std_male, train_std_female = return_tuple

    ####################
    # dev
    ####################
    print('---------- dev ----------')
    # Load target labels and save labels as npy files
    print('=> Processing ground truth labels...')
    label_dev_paths = prep.trans(data_type='dev')
    speaker_dict = read_sdb(label_paths=label_dev_paths,
                            label_type=label_type,
                            run_root_path=run_root_path,
                            save_path=label_dev_save_path)

    # Load htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(input_feature_path, htk_dir)
                 for htk_dir in sorted(glob(join(input_feature_path,
                                                 'dev/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_dev_save_path,
             speaker_dict=speaker_dict,
             normalize='speaker',
             is_training=False,
             train_mean_male=train_mean_male,
             train_std_male=train_std_male,
             train_mean_female=train_mean_female,
             train_std_female=train_std_female)

    ####################
    # eval1
    ####################
    print('---------- eval1 ----------')
    # Load target labels and save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval1_paths = prep.trans(data_type='eval1')
    speaker_dict = read_sdb(label_paths=label_eval1_paths,
                            label_type=label_type,
                            run_root_path=run_root_path,
                            is_test=True,
                            save_path=label_eval1_save_path)

    # Load htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(input_feature_path, htk_dir)
                 for htk_dir in sorted(glob(join(input_feature_path,
                                                 'eval1/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_eval1_save_path,
             speaker_dict=speaker_dict,
             normalize='speaker',
             is_training=False,
             train_mean_male=train_mean_male,
             train_std_male=train_std_male,
             train_mean_female=train_mean_female,
             train_std_female=train_std_female)

    ####################
    # eval2
    ####################
    print('---------- eval2 ----------')
    # Load target labels and save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval2_paths = prep.trans(data_type='eval2')
    speaker_dict = read_sdb(label_paths=label_eval2_paths,
                            label_type=label_type,
                            run_root_path=run_root_path,
                            is_test=True,
                            save_path=label_eval2_save_path)

    # Load htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(input_feature_path, htk_dir)
                 for htk_dir in sorted(glob(join(input_feature_path,
                                                 'eval2/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_eval2_save_path,
             speaker_dict=speaker_dict,
             normalize='speaker',
             is_training=False,
             train_mean_male=train_mean_male,
             train_std_male=train_std_male,
             train_mean_female=train_mean_female,
             train_std_female=train_std_female)

    ####################
    # eval3
    ####################
    print('---------- eval3 ----------')
    # Load target labels and save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval3_paths = prep.trans(data_type='eval3')
    speaker_dict = read_sdb(label_paths=label_eval3_paths,
                            label_type=label_type,
                            run_root_path=run_root_path,
                            is_test=True,
                            save_path=label_eval3_save_path)

    # Load htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(input_feature_path, htk_dir)
                 for htk_dir in sorted(glob(join(input_feature_path,
                                                 'eval3/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_eval3_save_path,
             speaker_dict=speaker_dict,
             normalize='speaker',
             is_training=False,
             train_mean_male=train_mean_male,
             train_std_male=train_std_male,
             train_mean_female=train_mean_female,
             train_std_female=train_std_female)

    # Make a confirmation file to prove that dataset was saved correctly
    with open(join(save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 5:
        raise ValueError

    csj_path = args[1]
    dataset_save_path = args[2]
    input_feature_path = args[3]
    run_root_path = args[4]

    for label_type in ['kanji', 'character', 'phone']:
        for train_data_type in ['large', 'default']:
            main(csj_path,
                 dataset_save_path,
                 input_feature_path,
                 run_root_path,
                 label_type,
                 train_data_type)
