#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC model (CSJ corpus)."""

import os
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

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(os.path.join(prep.fbank_path,
                                                         'train/*.htk')))]
    train_mean, train_std = read_htk(htk_paths=htk_paths,
                                     save_path=input_train_save_path,
                                     speaker_dict=speaker_dict,
                                     global_norm=False,
                                     is_training=True)

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

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(os.path.join(prep.fbank_path,
                                                         'train_plus/*.htk')))]
    train_mean2, train_std2 = read_htk(htk_paths=htk_paths,
                                       save_path=input_train_plus_save_path,
                                       speaker_dict=speaker_dict,
                                       global_norm=False,
                                       is_training=True)

    ####################
    # dev
    ####################
    print('---------- dev (19 speakers) ----------')
    print('=> copy for dev set (use dataset whose speaker_name starts "M" in default train set)')
    for input_path in glob(os.path.join(prep.dataset_path, 'ctc', label_type, 'train/input/M*')):
        print(input_path)
        speaker_name = os.path.basename(input_path)
        shutil.copytree(input_path, os.path.join(
            prep.dataset_path, 'ctc', label_type, 'dev/input', speaker_name))

    for label_path in glob(os.path.join(prep.dataset_path, 'ctc', label_type, 'train/label/M*')):
        print(label_path)
        speaker_name = os.path.basename(label_path)
        shutil.copytree(label_path, os.path.join(
            prep.dataset_path, 'ctc', label_type, 'dev/label', speaker_name))

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

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(os.path.join(prep.fbank_path,
                                                         'eval1/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_eval1_save_path,
             speaker_dict=speaker_dict,
             global_norm=False,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

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

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(os.path.join(prep.fbank_path,
                                                         'eval2/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_eval2_save_path,
             speaker_dict=speaker_dict,
             global_norm=False,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

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

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(os.path.join(prep.fbank_path,
                                                         'eval3/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_eval3_save_path,
             speaker_dict=speaker_dict,
             global_norm=False,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

    # make a confirmation file to prove that dataset was saved correctly
    with open(os.path.join(save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    print('=================================')
    print('=           CSJ (CTC)           =')
    print('=================================')

    label_types = ['character', 'phone']

    for label_type in label_types:
        main(label_type)
