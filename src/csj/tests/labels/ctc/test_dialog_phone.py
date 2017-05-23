#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

sys.path.append('../../../')
sys.path.append('../../../../')
from prepare_path import Prepare
from labels.ctc.dialog.character import read_sdb


class TestCTCLabelDialogPhone(unittest.TestCase):

    def test(self):
        prep = Prepare()
        label_dialog_core_paths = prep.trans(data_type='dialog_core')
        label_dialog_noncore_paths = prep.trans(data_type='dialog_noncore')

        print('===== dialog (phone) =====')
        for social_signal_type in ['insert', 'insert2', 'insert3', 'remove']:
            # core
            read_sdb(label_paths=label_dialog_core_paths,
                     label_type='phone',
                     speaker='L',
                     social_signal_type=social_signal_type)
            read_sdb(label_paths=label_dialog_noncore_paths,
                     label_type='phone',
                     speaker='R',
                     social_signal_type=social_signal_type)

            # noncore
            read_sdb(label_paths=label_dialog_noncore_paths,
                     label_type='phone',
                     speaker='L',
                     social_signal_type=social_signal_type)
            read_sdb(label_paths=label_dialog_noncore_paths,
                     label_type='phone',
                     speaker='R',
                     social_signal_type=social_signal_type)


if __name__ == '__main__':
    unittest.main()
