#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
from glob import glob

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.ctc.ldc97s62.character import read_trans


class TestInputGloablNormLDC97S62(unittest.TestCase):

    def test(self):
        prep = Prepare()

        label_paths = prep.label_train(label_type='character',
                                       train_type='ldc97s62')
        speaker_dict = read_trans(label_paths=label_paths)

        print('===== speaker norm (ldc97s62) =====')
        htk_paths = [os.path.join(prep.train_data_path, htk_dir)
                     for htk_dir in sorted(glob(os.path.join(prep.train_data_path, 'fbank/*.htk')))]
        read_htk(htk_paths=htk_paths,
                 speaker_dict=speaker_dict,
                 global_norm=False,
                 is_training=True)


if __name__ == '__main__':
    unittest.main()
