#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import glob
import numpy as np

sys.path.append('../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data_global_norm import read_htk
from labels.ctc.eval2000.swbd import read_transcript


class TestInputEval2000(unittest.TestCase):
    def test(self):
        prep = Prepare()

        label_test_paths = prep.label_test()
        speaker_dict = read_transcript(label_paths=label_test_paths)

        # load statistics over train dataset
        train_mean = np.load(os.path.join(prep.data_root_path,
                                          'dataset/ctc/character/train/train_mean.npy'))
        train_std = np.load(os.path.join(prep.data_root_path,
                                         'dataset/ctc/character/train/train_std.npy'))

        htk_test_paths = [os.path.join(prep.test_data_path, htk_dir)
                          for htk_dir in sorted(glob.glob(os.path.join(prep.test_data_path, 'fbank/swbd/*.htk')))]
        read_htk(htk_paths=htk_test_paths,
                 speaker_dict=speaker_dict,
                 normalize=True,
                 is_training=False,
                 train_mean=train_mean,
                 train_std=train_std)


if __name__ == '__main__':
    unittest.main()
