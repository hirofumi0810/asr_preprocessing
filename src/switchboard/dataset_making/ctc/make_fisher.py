#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC network (Fisher corpus)."""

import os
import sys
import shutil
from glob import glob
from tqdm import tqdm
import functools

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.ctc.fisher.character import read_trans
from utils.util import mkdir, join


def main(label_type):

    print('===== ' + label_type + ' =====')
    prep = Prepare()
    save_path = join(prep.dataset_path, 'ctc')
    save_path = join(save_path, label_type)
    save_path = join(save_path, 'train_fisher')

    # reset directory
    if not os.path.isfile(os.path.join(save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(save_path)):
            try:
                shutil.rmtree(os.path.join(save_path, c))
            except:
                os.remove(os.path.join(save_path, c))
    else:
        print('Already exists.')
        return 0

    input_save_path = join(save_path, 'input')
    label_save_path = join(save_path, 'label')

    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_train_paths = prep.label_train(label_type=label_type,
                                         train_type='fisher')
    speaker_dict_a = read_trans(label_paths=label_train_paths,
                                speaker='A',
                                save_path=label_save_path)
    speaker_dict_b = read_trans(label_paths=label_train_paths,
                                speaker='B',
                                save_path=label_save_path)

    # merge 2 dictionaries
    speaker_dict = functools.reduce(lambda first, second: dict(first, **second),
                                    [speaker_dict_a, speaker_dict_b])

    # read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_path = os.path.join(prep.train_data_path, 'fbank')
    train_mean, train_std = read_htk(htk_paths=htk_path,
                                     save_path=input_save_path,
                                     speaker_dict=speaker_dict,
                                     globa_norm=False,
                                     is_training=True)

    # make a confirmation file to prove that dataset was saved correctly
    with open(os.path.join(save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    print('========================================================')
    print('=                        Fisher                        =')
    print('========================================================')

    # TODO: add phone-level labels
    label_types = ['character']
    for label_type in label_types:
        main(label_type)
