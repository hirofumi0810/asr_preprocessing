#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import glob

sys.path.append('../')
sys.path.append('../../../')
from prepare_path import Prepare
from inputs.input_data_global_norm import read_htk
from labels.ctc.ldc97s62.character import read_transcript


class TestInputLDC97S62(unittest.TestCase):
    def test(self):
        prep = Prepare()

        label_train_paths = prep.label_train(label_type='character')
        speaker_dict = read_transcript(label_paths=label_train_paths)

        htk_train_paths = [os.path.join(prep.train_data_path, htk_dir)
                           for htk_dir in sorted(glob.glob(os.path.join(prep.train_data_path, 'fbank/*.htk')))]
        read_htk(htk_paths=htk_train_paths,
                 speaker_dict=speaker_dict,
                 normalize=True,
                 is_training=True)


if __name__ == '__main__':
    unittest.main()
