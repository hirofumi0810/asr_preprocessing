#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC network (Fisher corpus)."""

import os
import sys
import shutil
import glob
from tqdm import tqdm

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data_global_norm import read_htk
from labels.ctc.fisher.character import read_transcript as read_char
from utils.util import mkdir


def main(label_type, normalize_type):

    print('===== Fisher =====')
    prep = Prepare()
    save_path = mkdir(os.path.join(prep.dataset_path, 'ctc'))
    save_path = mkdir(os.path.join(save_path, label_type))

    # reset directory
    if not os.path.isfile(os.path.join(save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(save_path)):
            print(c)
            try:
                shutil.rmtree(os.path.join(save_path, c))
            except:
                os.remove(os.path.join(save_path, c))
    else:
        print('Already exists.')
        return 0

    train_save_path = mkdir(os.path.join(save_path, 'train'))

    input_train_save_path = mkdir(os.path.join(train_save_path, 'input'))
    label_train_save_path = mkdir(os.path.join(train_save_path, 'label'))

    print('-----' + label_type + '-----')
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_train_paths = prep.label_train_fisher(label_type=label_type)
    speaker_dict = read_char(label_paths=label_train_paths,
                             save_path=label_train_save_path)

    # read htk files, save input data as npy files, save frame num dict as a
    # pickle file
    print('=> Processing input data...')
    htk_train_path = os.path.join(prep.train_data_path, 'fbank')
    train_mean, train_std = read_htk(htk_paths=htk_train_path,
                                     save_path=input_train_save_path,
                                     speaker_dict=speaker_dict,
                                     normalize=True,
                                     is_training=True)

    htk_paths = [os.path.join(prep.train_data_fisher_path, htk_dir)
                 for htk_dir in sorted(glob.glob(os.path.join(prep.train_data_fisher_path, 'fbank/swbd/*.htk')))]
    read_htk(htk_paths=htk_paths,
             save_path=input_train_save_path,
             speaker_dict=speaker_dict,
             normalize=True,
             is_training=False,
             train_mean=train_mean,
             train_std=train_std)

    # make a confirmation file to prove that dataset was saved correctly
    with open(os.path.join(save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    print('========================================================')
    print('=                        Fisher                        =')
    print('========================================================')

    label_types = ['character']
    # normalize_types = ['global_norm', 'speaker_mean']
    normalize_types = ['global_norm']
    for normalize_type in normalize_types:
        for label_type in label_types:
            main(label_type, normalize_type)
