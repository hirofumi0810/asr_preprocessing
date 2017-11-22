#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys
import unittest
from glob import glob
from collections import Counter

sys.path.append('../../')
from swbd.labels.ldc97s62.character import read_trans as read_trans_swbd
from swbd.labels.fisher.character import read_trans as read_trans_fisher
from utils.measure_time_func import measure_time
from utils.util import mkdir_join

swbd_trans_path = '/n/sd8/inaguma/corpus/swbd/dataset/swb_ms98_transcriptions'
fisher_path = '/n/sd8/inaguma/corpus/swbd/data/fisher'

# Search paths to transcript
label_paths_swbd = []
for trans_path in glob(join(swbd_trans_path, '*/*/*.text')):
    if trans_path.split('.')[0][-5:] == 'trans':
        label_paths_swbd.append(trans_path)
label_paths_swbd = sorted(label_paths_swbd)

label_paths_fisher = []
for trans_path in glob(join(fisher_path, 'data/trans/*/*.txt')):
    label_paths_fisher.append(trans_path)
label_paths_fisher = sorted(label_paths_fisher)


class TestLabel2000h(unittest.TestCase):

    def test(self):

        self.check()

    @measure_time
    def check(self):

        speaker_dict_a, char_set_a, char_capital_set_a, word_count_dict_a = read_trans_fisher(
            label_paths=label_paths_fisher,
            target_speaker='A')
        speaker_dict_b, char_set_b, char_capital_set_b, word_count_dict_b = read_trans_fisher(
            label_paths=label_paths_fisher, target_speaker='B')

        # Meage 2 dictionaries
        speaker_dict_fisher = merge_dicts([speaker_dict_a, speaker_dict_b])
        char_set = char_set_a | char_set_b
        char_capital_set = char_capital_set_a | char_capital_set_b
        word_count_dict_fisher = dict(
            Counter(word_count_dict_a) + Counter(word_count_dict_b))

        read_trans_swbd(
            label_paths=label_paths_swbd,
            run_root_path='../',
            vocab_file_save_path=mkdir_join('../config/vocab_files'),
            save_vocab_file=True,
            speaker_dict_fisher=speaker_dict_fisher,
            char_set=char_set,
            char_capital_set=char_capital_set,
            word_count_dict=word_count_dict_fisher)


def merge_dicts(dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


if __name__ == '__main__':
    unittest.main()
