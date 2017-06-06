#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
from glob import glob
import numpy as np

sys.path.append('../../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data import read_htk
from labels.ctc.eval2000.swbd import read_txt


class TestInputGlobalNormEval2000(unittest.TestCase):

    def test(self):
        prep = Prepare()
        label_paths = prep.label_test('swbd')

        print('===== global norm (eval2000) =====')
        speaker_dict = read_txt(label_paths=label_paths)

        # Load statistics over train dataset
        train_mean = np.load(os.path.join(prep.data_path,
                                          'dataset/ctc/character/train/train_mean.npy'))
        train_std = np.load(os.path.join(prep.data_path,
                                         'dataset/ctc/character/train/train_std.npy'))

        htk_test_paths = [os.path.join(prep.test_data_path, htk_dir)
                          for htk_dir in sorted(glob(os.path.join(prep.test_data_path,
                                                                  'fbank/swbd/*.htk')))]
        read_htk(htk_paths=htk_test_paths,
                 speaker_dict=speaker_dict,
                 normalize='global',
                 is_training=False,
                 train_mean=train_mean,
                 train_std=train_std)


if __name__ == '__main__':
    unittest.main()
