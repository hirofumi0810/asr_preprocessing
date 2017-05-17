#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for CTC network (Switchboard corpus)."""

import os
import sys
import shutil
import glob
from tqdm import tqdm

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data_speaker_norm import read_htk as read_htk_speaker_norm
from inputs.input_data_global_norm import read_htk as read_htk_global_norm
from labels.ctc.ldc97s62.character import read_transcript as read_char
from labels.ctc.ldc97s62.phone import read_transcript as read_phone
# from labels.ctc.ldc97s62.word import read_trans as read_word
from utils.util import mkdir


def main(label_type, normalize_type):

    print('===== ' + label_type + ', ' + normalize_type + ' =====')
    prep = Prepare()
    save_path = mkdir(os.path.join(prep.dataset_path, 'ctc'))
    save_path = mkdir(os.path.join(save_path, label_type))

    train_save_path = mkdir(os.path.join(save_path, 'train'))
    # dev_save_path = mkdir(os.path.join(save_path, 'dev'))

    # reset directory
    if not os.path.isfile(os.path.join(train_save_path, 'check.txt')):
        print('=> Deleting old dataset...')
        for c in tqdm(os.listdir(train_save_path)):
            print(c)
            try:
                shutil.rmtree(os.path.join(train_save_path, c))
            except:
                os.remove(os.path.join(train_save_path, c))
    else:
        print('Already exists.')
        return 0

    input_train_save_path = mkdir(os.path.join(train_save_path, 'input'))
    label_train_save_path = mkdir(os.path.join(train_save_path, 'label'))
    # input_dev_save_path = mkdir(os.path.join(dev_save_path, 'input'))
    # label_dev_save_path = mkdir(os.path.join(dev_save_path, 'label'))

    ####################
    # train
    ####################
    # read labels, save labels as npy files
    print('=> Processing ground truth labels...')
    label_train_paths = prep.label_train(label_type=label_type)
    if label_type == 'character':
        speaker_dict = read_char(label_paths=label_train_paths,
                                 save_path=label_train_save_path)
    elif label_type == 'phone':
        speaker_dict = read_phone(label_paths=label_train_paths,
                                  save_path=label_train_save_path)

    # read htk files, save input data as npy files, save frame num dict as a
    # pickle file
    print('=> Processing input data...')
    htk_train_paths = [os.path.join(prep.train_data_path, htk_dir)
                       for htk_dir in sorted(glob.glob(os.path.join(prep.train_data_path, 'fbank/*.htk')))]
    if normalize_type == 'speaker_mean':
        # normalize per speaker
        read_htk_speaker_norm(htk_paths=htk_train_paths,
                              save_path=input_train_save_path,
                              speaker_dict=speaker_dict,
                              normalize=True)
    elif normalize_type == 'global_norm':
        # normalize over train set
        read_htk_global_norm(htk_paths=htk_train_paths,
                             save_path=input_train_save_path,
                             speaker_dict=speaker_dict,
                             normalize=True,
                             is_training=True)

    ####################
    # dev
    ####################
    # 最初の4000発話？

    # make a confirmation file to prove that dataset was saved correctly
    with open(os.path.join(train_save_path, 'check.txt'), 'w') as f:
        f.write('')
    print('Successfully completed!')


if __name__ == '__main__':

    print('============================================================')
    print('=                        LDC97S62                          =')
    print('============================================================')

    label_types = ['phone', 'character']
    # normalize_types = ['global_norm', 'speaker_mean']
    normalize_types = ['global_norm']
    for normalize_type in normalize_types:
        for label_type in label_types:
            main(label_type, normalize_type)
