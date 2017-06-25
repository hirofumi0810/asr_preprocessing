#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for the CTC model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import shutil
from glob import glob

sys.path.append('../')
from prepare_path import Prepare
from inputs.input_data import load_htk
from labels.ctc.character import load_sdb
from utils.util import mkdir_join


def main(csj_path, dataset_save_path, input_feature_path, run_root_path,
         label_type, train_data_type):

    print('===== ' + label_type + ' (' + train_data_type + ') =====')
    prep = Prepare(csj_path, run_root_path)
    input_save_path = mkdir_join(dataset_save_path, 'inputs')
    input_save_path = mkdir_join(input_save_path, train_data_type)
    label_save_path = mkdir_join(dataset_save_path, 'labels')
    label_save_path = mkdir_join(label_save_path, 'ctc')
    label_save_path = mkdir_join(label_save_path, label_type)
    label_save_path = mkdir_join(label_save_path, train_data_type)

    if train_data_type == 'default':
        train = 'train'
        save_map_file = False
    elif train_data_type == 'large':
        train = 'train_large'
        save_map_file = True

    ####################
    # labels
    ####################
    if not isfile(join(label_save_path, 'complete.txt')):
        print('=> Deleting old dataset...')
        shutil.rmtree(label_save_path)

        label_save_path = mkdir_join(dataset_save_path, 'labels')
        label_save_path = mkdir_join(label_save_path, 'ctc')
        label_save_path = mkdir_join(label_save_path, label_type)
        label_save_path = mkdir_join(label_save_path, train_data_type)

        label_train_save_path = mkdir_join(label_save_path, 'train')
        label_dev_save_path = mkdir_join(label_save_path, 'dev')
        label_eval1_save_path = mkdir_join(label_save_path, 'eval1')
        label_eval2_save_path = mkdir_join(label_save_path, 'eval2')
        label_eval3_save_path = mkdir_join(label_save_path, 'eval3')

        print('=> Processing transcripts...')
        # Load target labels and save labels as npy files
        print('---------- train ----------')
        label_train_paths = prep.trans(data_type=train)
        speaker_dict_train = load_sdb(label_paths=label_train_paths,
                                      label_type=label_type,
                                      run_root_path=run_root_path,
                                      save_path=label_train_save_path,
                                      save_map_file=save_map_file)

        print('---------- dev ----------')
        label_dev_paths = prep.trans(data_type='dev')
        speaker_dict_dev = load_sdb(label_paths=label_dev_paths,
                                    label_type=label_type,
                                    run_root_path=run_root_path,
                                    save_path=label_dev_save_path)

        print('---------- eval1 ----------')
        label_eval1_paths = prep.trans(data_type='eval1')
        speaker_dict_eval1 = load_sdb(label_paths=label_eval1_paths,
                                      label_type=label_type,
                                      run_root_path=run_root_path,
                                      is_test=True,
                                      save_path=label_eval1_save_path)

        print('---------- eval2 ----------')
        label_eval2_paths = prep.trans(data_type='eval2')
        speaker_dict_eval2 = load_sdb(label_paths=label_eval2_paths,
                                      label_type=label_type,
                                      run_root_path=run_root_path,
                                      is_test=True,
                                      save_path=label_eval2_save_path)

        print('---------- eval3 ----------')
        label_eval3_paths = prep.trans(data_type='eval3')
        speaker_dict_eval3 = load_sdb(label_paths=label_eval3_paths,
                                      label_type=label_type,
                                      run_root_path=run_root_path,
                                      is_test=True,
                                      save_path=label_eval3_save_path)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(label_save_path, 'complete.txt'), 'w') as f:
            f.write('')

    ####################
    # inputs
    ####################
    if not isfile(join(input_save_path, 'complete.txt')):
        print('=> Deleting old dataset...')
        shutil.rmtree(input_save_path)

        input_save_path = mkdir_join(dataset_save_path, 'inputs')
        input_save_path = mkdir_join(input_save_path, train_data_type)
        input_train_save_path = mkdir_join(input_save_path, 'train')
        input_dev_save_path = mkdir_join(input_save_path, 'dev')
        input_eval1_save_path = mkdir_join(input_save_path, 'eval1')
        input_eval2_save_path = mkdir_join(input_save_path, 'eval2')
        input_eval3_save_path = mkdir_join(input_save_path, 'eval3')

        print('=> Processing input data...')
        # Load htk files, and save input data and frame num dict
        htk_train_paths = [join(input_feature_path, htk_dir)
                           for htk_dir in sorted(glob(join(input_feature_path,
                                                           train, '*.htk')))]
        htk_dev_paths = [join(input_feature_path, htk_dir)
                         for htk_dir in sorted(glob(join(input_feature_path,
                                                         'dev/*.htk')))]
        htk_eval1_paths = [join(input_feature_path, htk_dir)
                           for htk_dir in sorted(glob(join(input_feature_path,
                                                           'eval1/*.htk')))]
        htk_eval2_paths = [join(input_feature_path, htk_dir)
                           for htk_dir in sorted(glob(join(input_feature_path,
                                                           'eval2/*.htk')))]
        htk_eval3_paths = [join(input_feature_path, htk_dir)
                           for htk_dir in sorted(glob(join(input_feature_path,
                                                           'eval3/*.htk')))]

        print('---------- train ----------')
        return_tuple = load_htk(htk_paths=htk_train_paths,
                                save_path=input_train_save_path,
                                speaker_dict=speaker_dict_train,
                                normalize='speaker',
                                is_training=True)
        train_mean_male = return_tuple[0]
        train_mean_female = return_tuple[1]
        train_std_male = return_tuple[2]
        train_std_female = return_tuple[3]

        print('---------- dev ----------')
        load_htk(htk_paths=htk_dev_paths,
                 save_path=input_dev_save_path,
                 speaker_dict=speaker_dict_dev,
                 normalize='speaker',
                 is_training=False,
                 train_mean_male=train_mean_male,
                 train_std_male=train_std_male,
                 train_mean_female=train_mean_female,
                 train_std_female=train_std_female)

        print('---------- eval1 ----------')
        load_htk(htk_paths=htk_eval1_paths,
                 save_path=input_eval1_save_path,
                 speaker_dict=speaker_dict_eval1,
                 normalize='speaker',
                 is_training=False,
                 train_mean_male=train_mean_male,
                 train_std_male=train_std_male,
                 train_mean_female=train_mean_female,
                 train_std_female=train_std_female)

        print('---------- eval2 ----------')
        load_htk(htk_paths=htk_eval2_paths,
                 save_path=input_eval2_save_path,
                 speaker_dict=speaker_dict_eval2,
                 normalize='speaker',
                 is_training=False,
                 train_mean_male=train_mean_male,
                 train_std_male=train_std_male,
                 train_mean_female=train_mean_female,
                 train_std_female=train_std_female)

        print('---------- eval3 ----------')
        load_htk(htk_paths=htk_eval3_paths,
                 save_path=input_eval3_save_path,
                 speaker_dict=speaker_dict_eval3,
                 normalize='speaker',
                 is_training=False,
                 train_mean_male=train_mean_male,
                 train_std_male=train_std_male,
                 train_mean_female=train_mean_female,
                 train_std_female=train_std_female)

        # Make a confirmation file to prove that dataset was saved correctly
        with open(join(input_save_path, 'check.txt'), 'w') as f:
            f.write('')


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
