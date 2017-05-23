#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC model (CSJ corpus, dialog)."""

import os
import sys
import shutil
from glob import glob
from tqdm import tqdm
import shutil
import functools


sys.path.append('../')
sys.path.append('../../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.ctc.dialog.character import read_sdb
from utils.util import mkdir


def main(label_type, social_signal_type):

    print('===== ' + label_type + ', ' + social_signal_type + ' =====')
    prep = Prepare()
    save_path = mkdir(os.path.join(prep.dataset_path, 'dialog'))
    save_path = mkdir(os.path.join(save_path, 'ctc'))
    save_path = mkdir(os.path.join(save_path, label_type))
    save_path = mkdir(os.path.join(save_path, social_signal_type))

    # reset directory
    if not os.path.isfile(os.path.join(save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        try:
            for c in tqdm(os.listdir(save_path)):
                shutil.rmtree(os.path.join(save_path, c))
        except:
            os.remove(os.path.join(save_path, c))
    else:
        print('Already exists.')
        return 0

    input_save_path = mkdir(os.path.join(save_path, 'input'))
    label_save_path = mkdir(os.path.join(save_path, 'label'))

    ######################
    # core dialog (ver4)
    ######################
    print('---------- core dialog ----------')
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_paths = prep.trans(data_type='dialog_core')
    speaker_dict_left = read_sdb(label_paths=label_paths,
                                 label_type=label_type,
                                 speaker='L',
                                 social_signal_type=social_signal_type,
                                 save_path=label_save_path)
    speaker_dict_right = read_sdb(label_paths=label_paths,
                                  label_type=label_type,
                                  speaker='R',
                                  social_signal_type=social_signal_type,
                                  save_path=label_save_path)

    # merge 2 dict
    speaker_dict = functools.reduce(lambda first, second: dict(first, **second),
                                    [speaker_dict_left, speaker_dict_right])

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_paths = [os.path.join(prep.fbank_path, htk_dir)
                 for htk_dir in sorted(glob(os.path.join(prep.fbank_path,
                                                         'dialog_core/*.htk')))]
    train_mean, train_std = read_htk(htk_paths=htk_paths,
                                     save_path=input_save_path,
                                     speaker_dict=speaker_dict,
                                     global_norm=False,
                                     is_training=True)

    # make a confirmation file to prove that dataset was saved correctly
    with open(os.path.join(save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    print('=================================================')
    print('=            CSJ for dialog (CTC)               =')
    print('=================================================')

    label_types = ['character', 'phone']
    social_signal_types = ['insert', 'insert2', 'insert3', 'remove']
    for label_type in label_types:
        for social_signal_type in social_signal_types:
            main(label_type, social_signal_type)
