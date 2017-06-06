#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC model (CSJ corpus, monolog)."""

import os
from os.path import join
import sys
import shutil
from glob import glob
from tqdm import tqdm
import shutil


sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.ctc.monolog.character import read_sdb
from utils.util import mkdir, mkdir_join


def main(label_type, train_data_type):

    print('===== ' + label_type + ' (' + train_data_type + ') =====')
    prep = Prepare()
    save_path = mkdir_join(prep.dataset_path, 'monolog')
    save_path = mkdir_join(save_path, 'ctc')
    save_path = mkdir_join(save_path, label_type)
    save_path = mkdir_join(save_path, train_data_type)

    # reset directory
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
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    if train_data_type == 'default':
        data_type = 'train'
        save_map_file = False
    elif train_data_type == 'large':
        data_type = 'train_all'
        save_map_file = True
    label_train_paths = prep.trans(data_type=data_type)
    speaker_dict = read_sdb(label_paths=label_train_paths,
                            label_type=label_type,
                            save_path=label_train_save_path,
                            save_map_file=save_map_file)

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(join(prep.fbank_path,
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
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_dev_paths = prep.trans(data_type='dev')
    speaker_dict = read_sdb(label_paths=label_dev_paths,
                            label_type=label_type,
                            save_path=label_dev_save_path)

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(join(prep.fbank_path,
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
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval1_paths = prep.trans(data_type='eval1')
    speaker_dict = read_sdb(label_paths=label_eval1_paths,
                            label_type=label_type,
                            is_test=True,
                            save_path=label_eval1_save_path)

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(join(prep.fbank_path,
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
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval2_paths = prep.trans(data_type='eval2')
    speaker_dict = read_sdb(label_paths=label_eval2_paths,
                            label_type=label_type,
                            is_test=True,
                            save_path=label_eval2_save_path)

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(join(prep.fbank_path,
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
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_eval3_paths = prep.trans(data_type='eval3')
    speaker_dict = read_sdb(label_paths=label_eval3_paths,
                            label_type=label_type,
                            is_test=True,
                            save_path=label_eval3_save_path)

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(join(prep.fbank_path,
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

    # make a confirmation file to prove that dataset was saved correctly
    with open(join(save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    print('=====================================================')
    print('=               CSJ for monolog (CTC)               =')
    print('=====================================================')

    for label_type in ['kanji', 'character', 'phone']:
        for train_data_type in ['large', 'default']:
            main(label_type, train_data_type)
