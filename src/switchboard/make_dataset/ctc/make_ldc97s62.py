#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC network (Switchboard corpus)."""

import os
from os.path import join
import sys
import shutil
from glob import glob
from tqdm import tqdm

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data import read_htk as read_htk
from labels.ctc.ldc97s62.character import read_trans as read_char
from labels.ctc.ldc97s62.phone import read_trans as read_phone
# from labels.ctc.ldc97s62.word import read_word as read_word
from utils.util import mkdir_join


def main(label_type):

    print('===== ' + label_type + ' =====')
    prep = Prepare()
    save_path = mkdir_join(prep.dataset_path, 'ctc')
    save_path = mkdir_join(save_path, label_type)

    train_save_path = mkdir_join(save_path, 'train')
    # dev_save_path = mkdir_join(save_path, 'dev')

    # Reset directory
    if not os.path.isfile(join(train_save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(train_save_path)):
            try:
                shutil.rmtree(join(train_save_path, c))
            except:
                os.remove(join(train_save_path, c))
    else:
        print('Already exists.')
        return 0

    input_train_save_path = mkdir_join(train_save_path, 'input')
    label_train_save_path = mkdir_join(train_save_path, 'label')
    # input_dev_save_path = mkdir_join(dev_save_path, 'input')
    # label_dev_save_path = mkdir_join(dev_save_path, 'label')

    ####################
    # train
    ####################
    # Read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_train_paths = prep.label_train(label_type=label_type)
    if label_type == 'character':
        speaker_dict = read_char(label_paths=label_train_paths,
                                 save_path=label_train_save_path)
    elif label_type == 'phone':
        speaker_dict = read_phone(label_paths=label_train_paths,
                                  save_path=label_train_save_path)

    # Read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [join(prep.train_data_path, htk_dir)
                 for htk_dir in sorted(glob(join(prep.train_data_path,
                                                 'fbank/*.htk')))]

    # Normalize over train set
    train_mean, train_std = read_htk(htk_paths=htk_paths,
                                     save_path=input_train_save_path,
                                     speaker_dict=speaker_dict,
                                     normalize='speaker',
                                     is_training=True)

    # TODO: Merge statistics of ldc97s62 and fisher

    ####################
    # dev
    ####################
    # 最初の4000発話？

    # Make a confirmation file to prove that dataset was saved correctly
    with open(join(train_save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    print('============================================================')
    print('=                        LDC97S62                          =')
    print('============================================================')

    for label_type in ['phone', 'character']:
        main(label_type)
