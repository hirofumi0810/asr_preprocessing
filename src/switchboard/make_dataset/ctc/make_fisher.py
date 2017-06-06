#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC network (Fisher corpus)."""

import os
from os.path import join
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
from utils.util import mkdir_join


def main(label_type):

    print('===== ' + label_type + ' =====')
    prep = Prepare()
    save_path = mkdir_join(prep.dataset_path, 'ctc')
    save_path = mkdir_join(save_path, label_type)
    save_path = mkdir_join(save_path, 'train_fisher')

    # Reset directory
    if not os.path.isfile(join(save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(save_path)):
            try:
                shutil.rmtree(join(save_path, c))
            except:
                os.remove(join(save_path, c))
    else:
        print('Already exists.')
        return 0

    input_save_path = mkdir_join(save_path, 'input')
    label_save_path = mkdir_join(save_path, 'label')

    # Read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_train_paths = prep.label_train(label_type=label_type,
                                         train_type='fisher')
    print('----- Speaker A -----')
    speaker_dict_a = read_trans(label_paths=label_train_paths,
                                speaker='A',
                                save_path=label_save_path)
    print('----- Speaker B -----')
    speaker_dict_b = read_trans(label_paths=label_train_paths,
                                speaker='B',
                                save_path=label_save_path)

    # Merge 2 dictionaries
    speaker_dict = functools.reduce(lambda first, second: dict(first, **second),
                                    [speaker_dict_a, speaker_dict_b])

    # Read htk files, save input data & frame num dict
    print('=> Processing input data...')
    htk_path = join(prep.train_data_path, 'fbank')
    train_mean, train_std = read_htk(htk_paths=htk_path,
                                     save_path=input_save_path,
                                     speaker_dict=speaker_dict,
                                     normalize='speaker',
                                     is_training=True)

    # TODO: Merge statistics of ldc97s62 and fisher

    # Make a confirmation file to prove that dataset was saved correctly
    with open(join(save_path, 'check.txt'), 'w') as f:
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
