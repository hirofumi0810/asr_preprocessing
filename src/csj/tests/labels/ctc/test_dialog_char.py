#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../../../')
sys.path.append('../../../../')
from prepare_path import Prepare
from labels.ctc.dialog.character import read_sdb


class TestCTCLabelDialogChar(unittest.TestCase):

    def test(self):
        prep = Prepare()
        label_dialog_train_paths = prep.trans(data_type='dialog_train')
        label_dialog_dev_paths = prep.trans(data_type='dialog_dev')
        label_dialog_test_paths = prep.trans(data_type='dialog_test')
        label_dialog_train_paths = set(label_dialog_train_paths)
        for path in label_dialog_test_paths:
            label_dialog_train_paths.remove(path)
        label_dialog_train_paths = list(label_dialog_train_paths)

        print('===== dialog (character) =====')
        for social_signal_type in ['insert', 'insert2', 'insert3', 'remove']:
            # train
            read_sdb(label_paths=label_dialog_train_paths,
                     label_type='character',
                     speaker='L',
                     social_signal_type=social_signal_type)
            read_sdb(label_paths=label_dialog_train_paths,
                     label_type='character',
                     speaker='R',
                     social_signal_type=social_signal_type)
            read_sdb(label_paths=label_dialog_test_paths,
                     label_type='character',
                     speaker='L',
                     social_signal_type=social_signal_type)

            # dev
            read_sdb(label_paths=label_dialog_dev_paths,
                     label_type='character',
                     speaker='R',
                     social_signal_type=social_signal_type)

            # test
            read_sdb(label_paths=label_dialog_test_paths,
                     label_type='character',
                     speaker='R',
                     social_signal_type=social_signal_type)


if __name__ == '__main__':
    unittest.main()
